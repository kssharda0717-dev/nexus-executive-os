"""MCP Executive Bridge — Outlook + Slack integration for NEXUS.

This module implements the Model Context Protocol (MCP) bridge that connects
NEXUS's local agent mesh to external enterprise systems:

    Outlook (Microsoft Graph API):
        - Fetch unread emails
        - Read Outlook Calendar invites
        - Accept/Decline invites programmatically
        - Detect conflicts between Outlook invites and NEXUS calendar

    Slack (Slack Web API + Events API):
        - Send formatted conflict reports
        - Interactive Approve/Decline blocks
        - Process user responses back to the agent mesh

Architecture:
    This is a MODULAR bridge. It is imported lazily by the agent system.
    If MICROSOFT_GRAPH_CLIENT_ID, MICROSOFT_GRAPH_CLIENT_SECRET, SLACK_BOT_TOKEN,
    or SLACK_SIGNING_SECRET are missing from the environment, the bridge
    initializes in DISABLED mode — all methods return graceful no-ops.
    The core NEXUS dashboard and local agents are NEVER affected.

MCP Pattern:
    The bridge exposes two "tool" classes (OutlookManager, SlackMessenger)
    that follow the MCP tool interface: each method is a self-contained
    action with typed inputs, structured outputs, and error boundaries.
    The PrimaryAgent can call these just like local tools.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("nexus.mcp_executive")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Graceful when env vars are missing
# ═══════════════════════════════════════════════════════════════

@dataclass
class MCPConfig:
    """MCP Executive Bridge configuration.

    All credentials come from environment variables.
    If any are missing, the corresponding service operates in DISABLED mode.
    """
    # Microsoft Graph (Outlook)
    ms_client_id: str = ""
    ms_client_secret: str = ""
    ms_tenant_id: str = "common"
    ms_redirect_uri: str = "http://localhost:8000/auth/microsoft/callback"

    # Slack
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_channel: str = "#nexus-notifications"

    # State
    ms_access_token: str = ""
    ms_refresh_token: str = ""
    ms_token_expiry: float = 0.0

    @property
    def outlook_enabled(self) -> bool:
        return bool(self.ms_client_id and self.ms_client_secret)

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_bot_token)

    @property
    def fully_enabled(self) -> bool:
        return self.outlook_enabled and self.slack_enabled


def load_mcp_config() -> MCPConfig:
    """Load MCP configuration from environment. Never fails."""
    return MCPConfig(
        ms_client_id=os.getenv("MICROSOFT_GRAPH_CLIENT_ID", ""),
        ms_client_secret=os.getenv("MICROSOFT_GRAPH_CLIENT_SECRET", ""),
        ms_tenant_id=os.getenv("MICROSOFT_GRAPH_TENANT_ID", "common"),
        ms_redirect_uri=os.getenv("MICROSOFT_GRAPH_REDIRECT_URI",
                                  "http://localhost:8000/auth/microsoft/callback"),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
        slack_channel=os.getenv("SLACK_NEXUS_CHANNEL", "#nexus-notifications"),
    )


# ═══════════════════════════════════════════════════════════════
# OUTLOOK MANAGER — Microsoft Graph API MCP Tool
# ═══════════════════════════════════════════════════════════════

# ── Per-User Token Resolution ─────────────────────────────
# The bridge resolves tokens from the integrations DB when a user_id
# is available, falling back to the global .env config for backwards
# compatibility (single-user / demo mode).

INTEGRATION_REQUIRED_ERROR = {
    "status": "integration_required",
    "code": "INTEGRATION_REQUIRED",
    "message": "This service is not connected. Please go to Settings > Integrations and connect your account.",
}


async def _resolve_user_token(user_id: str, provider: str) -> dict | None:
    """Fetch and decrypt a user's integration tokens from the DB.

    Returns: {"access_token": str, "refresh_token": str, "token_expiry": str, "extra_data": dict}
    or None if no active integration exists.
    """
    try:
        import database as db_module
        from security import decrypt_token
        integration = await db_module.get_integration(user_id, provider)
        if not integration:
            return None
        return {
            "access_token": decrypt_token(integration.get("access_token", "")),
            "refresh_token": decrypt_token(integration.get("refresh_token", "")),
            "token_expiry": integration.get("token_expiry", ""),
            "scopes": integration.get("scopes", ""),
            "extra_data": integration.get("extra_data", {}),
        }
    except Exception as e:
        logger.error(f"Failed to resolve token for user={user_id} provider={provider}: {e}")
        return None


class OutlookManager:
    """MCP Tool: Outlook email and calendar integration via Microsoft Graph.

    Multi-Tenant: When user_id is provided, fetches per-user tokens from the
    integrations table. Falls back to the global MCPConfig for demo/single-user mode.

    Methods follow the MCP tool pattern:
        - Typed inputs (dataclass or dict)
        - Structured JSON outputs
        - Built-in error boundaries (never raises to caller)
    """

    GRAPH_BASE = "https://graph.microsoft.com/v1.0"
    SCOPES = ["Mail.Read", "Calendars.ReadWrite", "User.Read"]

    def __init__(self, config: MCPConfig):
        self._config = config
        self._http = None  # Lazy httpx.AsyncClient

    @property
    def enabled(self) -> bool:
        return self._config.outlook_enabled

    async def _get_http(self):
        """Lazy-init async HTTP client."""
        if self._http is None:
            try:
                import httpx
                self._http = httpx.AsyncClient(timeout=30.0)
            except ImportError:
                logger.warning("httpx not installed — Outlook integration disabled")
                return None
        return self._http

    async def _get_headers(self, user_id: str = None) -> dict:
        """Get auth headers, trying per-user token first, then global fallback."""
        # Per-user token resolution (multi-tenant)
        if user_id:
            user_tokens = await _resolve_user_token(user_id, "outlook")
            if user_tokens and user_tokens.get("access_token"):
                return {
                    "Authorization": f"Bearer {user_tokens['access_token']}",
                    "Content-Type": "application/json",
                }

        # Global fallback (single-user / demo mode)
        if not self._config.ms_access_token:
            return {}
        return {
            "Authorization": f"Bearer {self._config.ms_access_token}",
            "Content-Type": "application/json",
        }

    def get_auth_url(self) -> str:
        """Generate the OAuth2 authorization URL for Microsoft login."""
        if not self.enabled:
            return ""
        base = f"https://login.microsoftonline.com/{self._config.ms_tenant_id}/oauth2/v2.0/authorize"
        params = {
            "client_id": self._config.ms_client_id,
            "response_type": "code",
            "redirect_uri": self._config.ms_redirect_uri,
            "scope": " ".join(self.SCOPES),
            "response_mode": "query",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{query}"

    async def exchange_code(self, auth_code: str) -> dict:
        """Exchange an OAuth2 authorization code for access + refresh tokens."""
        if not self.enabled:
            return {"error": "Outlook integration not configured"}

        http = await self._get_http()
        if not http:
            return {"error": "HTTP client unavailable"}

        try:
            token_url = f"https://login.microsoftonline.com/{self._config.ms_tenant_id}/oauth2/v2.0/token"
            resp = await http.post(token_url, data={
                "client_id": self._config.ms_client_id,
                "client_secret": self._config.ms_client_secret,
                "code": auth_code,
                "redirect_uri": self._config.ms_redirect_uri,
                "grant_type": "authorization_code",
                "scope": " ".join(self.SCOPES),
            })
            data = resp.json()
            if "access_token" in data:
                self._config.ms_access_token = data["access_token"]
                self._config.ms_refresh_token = data.get("refresh_token", "")
                self._config.ms_token_expiry = time.time() + data.get("expires_in", 3600)
                logger.info("Microsoft Graph OAuth2 tokens acquired")
                return {"status": "authenticated", "expires_in": data.get("expires_in", 3600)}
            return {"error": data.get("error_description", "Token exchange failed")}
        except Exception as e:
            logger.error(f"OAuth2 token exchange failed: {e}")
            return {"error": str(e)}

    async def fetch_unread_emails(self, top: int = 10, user_id: str = None) -> dict:
        """Fetch unread emails from Outlook inbox.

        Returns: {emails: [{subject, from, received, preview, id}], count: int}
        """
        if not self.enabled:
            return {"emails": [], "count": 0, "status": "disabled",
                    "message": "Outlook integration not configured. Set MICROSOFT_GRAPH_CLIENT_ID and MICROSOFT_GRAPH_CLIENT_SECRET."}

        http = await self._get_http()
        headers = await self._get_headers(user_id=user_id)
        if not http or not headers:
            return {"emails": [], "count": 0, "status": "unauthenticated",
                    "message": "Please connect your Microsoft Outlook account via the Neural Link section first."}

        try:
            resp = await http.get(
                f"{self.GRAPH_BASE}/me/messages",
                headers=headers,
                params={
                    "$filter": "isRead eq false",
                    "$top": top,
                    "$select": "subject,from,receivedDateTime,bodyPreview,id",
                    "$orderby": "receivedDateTime desc",
                },
            )
            data = resp.json()
            emails = []
            for msg in data.get("value", []):
                emails.append({
                    "id": msg.get("id", ""),
                    "subject": msg.get("subject", "(No Subject)"),
                    "from": msg.get("from", {}).get("emailAddress", {}).get("address", "unknown"),
                    "received": msg.get("receivedDateTime", ""),
                    "preview": msg.get("bodyPreview", "")[:200],
                })
            return {"emails": emails, "count": len(emails), "status": "ok"}
        except Exception as e:
            logger.error(f"Failed to fetch emails: {e}")
            return {"emails": [], "count": 0, "status": "error", "message": str(e)}

    async def fetch_calendar_events(self, start: str = None, end: str = None,
                                    user_id: str = None) -> dict:
        """Fetch Outlook calendar events for a date range.

        Returns: {events: [{subject, start, end, organizer, id}], count: int}
        """
        if not self.enabled:
            return {"events": [], "count": 0, "status": "disabled"}

        http = await self._get_http()
        headers = await self._get_headers(user_id=user_id)
        if not http or not headers:
            return {"events": [], "count": 0, "status": "unauthenticated",
                    "message": "Please connect your Microsoft Outlook account via the Neural Link section first."}

        # Default: next 7 days
        if not start:
            start = datetime.now().isoformat()
        if not end:
            end = (datetime.now() + timedelta(days=7)).isoformat()

        try:
            resp = await http.get(
                f"{self.GRAPH_BASE}/me/calendarview",
                headers=headers,
                params={
                    "startDateTime": start,
                    "endDateTime": end,
                    "$select": "subject,start,end,organizer,id,isCancelled",
                    "$orderby": "start/dateTime",
                    "$top": 50,
                },
            )
            data = resp.json()
            events = []
            for evt in data.get("value", []):
                if evt.get("isCancelled"):
                    continue
                events.append({
                    "id": evt.get("id", ""),
                    "subject": evt.get("subject", "(No Subject)"),
                    "start": evt.get("start", {}).get("dateTime", ""),
                    "end": evt.get("end", {}).get("dateTime", ""),
                    "organizer": evt.get("organizer", {}).get("emailAddress", {}).get("name", ""),
                })
            return {"events": events, "count": len(events), "status": "ok"}
        except Exception as e:
            logger.error(f"Failed to fetch Outlook calendar: {e}")
            return {"events": [], "count": 0, "status": "error", "message": str(e)}

    async def respond_to_invite(self, event_id: str, action: str, comment: str = "",
                               user_id: str = None) -> dict:
        """Accept or decline an Outlook calendar invite.

        Args:
            event_id: The Outlook event ID
            action: "accept" or "decline"
            comment: Optional response message
            user_id: Optional user_id for per-user token resolution
        """
        if not self.enabled:
            return {"status": "disabled"}

        http = await self._get_http()
        headers = await self._get_headers(user_id=user_id)
        if not http or not headers:
            return {"status": "unauthenticated",
                    "message": "Please connect your Microsoft Outlook account via the Neural Link section first."}

        endpoint = "accept" if action.lower() == "accept" else "decline"
        try:
            resp = await http.post(
                f"{self.GRAPH_BASE}/me/events/{event_id}/{endpoint}",
                headers=headers,
                json={"comment": comment or f"Responded via NEXUS: {action}"},
            )
            if resp.status_code in (200, 202):
                return {"status": "ok", "action": endpoint, "event_id": event_id}
            return {"status": "error", "code": resp.status_code, "detail": resp.text[:200]}
        except Exception as e:
            logger.error(f"Failed to {endpoint} invite: {e}")
            return {"status": "error", "message": str(e)}

    async def detect_conflicts(self, invite_start: str, invite_end: str) -> dict:
        """Check for conflicts between an Outlook invite and the NEXUS local calendar.

        Returns both Outlook conflicts and NEXUS local conflicts.
        """
        import database as db_module

        conflicts = {"outlook": [], "nexus": [], "has_conflict": False}

        # Check NEXUS local calendar
        try:
            local_events = await db_module.list_events(invite_start, invite_end)
            for evt in local_events:
                conflicts["nexus"].append({
                    "title": evt.get("title", ""),
                    "start": evt.get("start_time", ""),
                    "end": evt.get("end_time", ""),
                })
        except Exception as e:
            logger.warning(f"Could not check NEXUS calendar: {e}")

        # Check Outlook calendar
        if self.enabled and self._config.ms_access_token:
            outlook_result = await self.fetch_calendar_events(invite_start, invite_end)
            for evt in outlook_result.get("events", []):
                conflicts["outlook"].append({
                    "subject": evt.get("subject", ""),
                    "start": evt.get("start", ""),
                    "end": evt.get("end", ""),
                })

        conflicts["has_conflict"] = bool(conflicts["outlook"] or conflicts["nexus"])
        return conflicts


# ═══════════════════════════════════════════════════════════════
# SLACK MESSENGER — Slack Web API MCP Tool
# ═══════════════════════════════════════════════════════════════

class SlackMessenger:
    """MCP Tool: Slack messaging with interactive blocks.

    Sends formatted conflict reports and processes user responses
    (Approve/Decline) back through the NEXUS agent mesh.
    """

    SLACK_API = "https://slack.com/api"

    def __init__(self, config: MCPConfig):
        self._config = config
        self._http = None
        self._pending_actions: dict[str, dict] = {}  # message_ts → action context

    @property
    def enabled(self) -> bool:
        return self._config.slack_enabled

    async def _get_http(self):
        if self._http is None:
            try:
                import httpx
                self._http = httpx.AsyncClient(timeout=15.0)
            except ImportError:
                logger.warning("httpx not installed — Slack integration disabled")
                return None
        return self._http

    def _headers(self, token_override: str = None) -> dict:
        token = token_override or self._config.slack_bot_token
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def _resolve_slack_token(self, user_id: str = None) -> str:
        """Resolve Slack bot token: per-user DB first, then global .env fallback."""
        if user_id:
            user_tokens = await _resolve_user_token(user_id, "slack")
            if user_tokens and user_tokens.get("access_token"):
                return user_tokens["access_token"]
        return self._config.slack_bot_token

    def _text_to_blocks(self, text: str) -> list[dict]:
        """Convert plain/Markdown text into Slack Block Kit blocks.

        Produces professional-looking Slack messages with sections,
        headers, and dividers instead of raw Markdown stars/hashes.
        """
        blocks = []
        lines = text.split('\n')
        current_section = []

        def flush_section():
            if current_section:
                mrkdwn_text = '\n'.join(current_section).strip()
                if mrkdwn_text:
                    # Convert markdown bold/italic to Slack mrkdwn
                    mrkdwn_text = re.sub(r'\*\*\*(.+?)\*\*\*', r'*_\1_*', mrkdwn_text)
                    mrkdwn_text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', mrkdwn_text)
                    blocks.append({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": mrkdwn_text[:3000]}
                    })
                current_section.clear()

        for line in lines:
            stripped = line.strip()

            # Horizontal rules → divider
            if re.match(r'^-{3,}$|^\*{3,}$|^_{3,}$', stripped):
                flush_section()
                blocks.append({"type": "divider"})
                continue

            # H1/H2 headers → header block
            h_match = re.match(r'^#{1,2}\s+(.+)', stripped)
            if h_match:
                flush_section()
                blocks.append({
                    "type": "header",
                    "text": {"type": "plain_text", "text": h_match.group(1)[:150], "emoji": True}
                })
                continue

            # H3+ headers → bold section
            h3_match = re.match(r'^#{3,}\s+(.+)', stripped)
            if h3_match:
                flush_section()
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{h3_match.group(1)}*"}
                })
                continue

            # Everything else accumulates into current section
            current_section.append(line)

        flush_section()

        # Always add NEXUS context footer
        blocks.append({
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "🤖 _Sent by NEXUS Autonomous Life OS_"}
            ]
        })

        return blocks

    async def send_message(self, text: str, channel: str = None,
                           blocks: list = None, file_path: str = None,
                           file_title: str = None, user_id: str = None) -> dict:
        """Send a message to a Slack channel with Block Kit formatting.

        Args:
            text: Fallback text (shown in notifications)
            channel: Slack channel (defaults to config)
            blocks: Optional Block Kit blocks (auto-generated from text if None)
            file_path: Optional file path to upload as attachment
            file_title: Optional title for the uploaded file
            user_id: Optional user_id for per-user token resolution
        """
        if not self.enabled:
            return {"status": "disabled",
                    "message": "Slack not configured. Set SLACK_BOT_TOKEN."}

        http = await self._get_http()
        if not http:
            return {"status": "error", "message": "HTTP client unavailable"}

        # Resolve per-user or global token
        token = await self._resolve_slack_token(user_id)
        if not token:
            return {**INTEGRATION_REQUIRED_ERROR, "provider": "slack"}

        channel = channel or self._config.slack_channel

        try:
            # ── File upload flow ──
            if file_path and os.path.isfile(file_path):
                return await self._upload_file(
                    http, channel, file_path,
                    file_title or os.path.basename(file_path),
                    text, token=token,
                )

            # ── Message flow with auto Block Kit ──
            # Auto-convert plain text to Block Kit if no blocks provided
            if not blocks and len(text) > 100:
                blocks = self._text_to_blocks(text)

            payload = {"channel": channel, "text": text}
            if blocks:
                payload["blocks"] = blocks

            resp = await http.post(
                f"{self.SLACK_API}/chat.postMessage",
                headers=self._headers(token_override=token),
                json=payload,
            )
            data = resp.json()
            if data.get("ok"):
                return {"status": "ok", "channel": channel,
                        "ts": data.get("ts", ""), "message_id": data.get("ts", "")}
            return {"status": "error", "detail": data.get("error", "unknown")}
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _upload_file(self, http, channel: str, file_path: str,
                           title: str, initial_comment: str = "",
                           token: str = None) -> dict:
        """Upload a file to Slack using files.upload API.

        Args:
            http: httpx.AsyncClient
            channel: Target channel
            file_path: Local file path
            title: Display title for the file
            initial_comment: Message text accompanying the upload
            token: Slack bot token (per-user or global)
        """
        import mimetypes
        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        filename = os.path.basename(file_path)
        auth_token = token or self._config.slack_bot_token

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Slack files.upload v1 — reliable for files under 50MB
            resp = await http.post(
                f"{self.SLACK_API}/files.upload",
                headers={"Authorization": f"Bearer {auth_token}"},
                data={
                    "channels": channel,
                    "title": title,
                    "filename": filename,
                    "initial_comment": initial_comment or f"📎 Here is your file: *{title}*",
                },
                files={"file": (filename, file_data, content_type)},
            )
            data = resp.json()
            if data.get("ok"):
                file_info = data.get("file", {})
                return {
                    "status": "ok",
                    "channel": channel,
                    "file_id": file_info.get("id", ""),
                    "file_url": file_info.get("permalink", ""),
                    "message": f"File uploaded: {title}",
                }
            return {"status": "error", "detail": data.get("error", "upload_failed")}
        except Exception as e:
            logger.error(f"Slack file upload failed: {e}")
            return {"status": "error", "message": str(e)}

    async def send_conflict_report(self, invite_subject: str, invite_start: str,
                                   invite_end: str, conflicts: dict,
                                   outlook_event_id: str = "",
                                   channel: str = None) -> dict:
        """Send an interactive conflict report to Slack with Approve/Decline buttons.

        This is the core of the "Outlook-Slack Loop" — the user sees the conflict
        and decides whether to accept or decline the Outlook invite.
        """
        if not self.enabled:
            return {"status": "disabled"}

        # Build conflict text
        conflict_lines = []
        for c in conflicts.get("nexus", []):
            conflict_lines.append(f"• NEXUS: _{c['title']}_ ({c['start']} → {c['end']})")
        for c in conflicts.get("outlook", []):
            conflict_lines.append(f"• Outlook: _{c['subject']}_ ({c['start']} → {c['end']})")

        conflict_text = "\n".join(conflict_lines) if conflict_lines else "_No conflicts detected_"
        has_conflict = conflicts.get("has_conflict", False)

        # Block Kit layout
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "📅 New Calendar Invite", "emoji": True}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Subject:*\n{invite_subject}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{invite_start} → {invite_end}"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{'⚠️ *Conflicts Detected:*' if has_conflict else '✅ *No Conflicts:*'}\n{conflict_text}"
                }
            },
            {"type": "divider"},
            {
                "type": "actions",
                "block_id": "invite_response",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Accept", "emoji": True},
                        "style": "primary",
                        "action_id": "accept_invite",
                        "value": outlook_event_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Decline", "emoji": True},
                        "style": "danger",
                        "action_id": "decline_invite",
                        "value": outlook_event_id,
                    },
                ]
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": "🤖 _Sent by NEXUS Executive Bridge_"}
                ]
            },
        ]

        result = await self.send_message(
            text=f"New invite: {invite_subject} — {'⚠️ CONFLICT' if has_conflict else '✅ Clear'}",
            channel=channel,
            blocks=blocks,
        )

        # Track pending action for response loop
        if result.get("ts"):
            self._pending_actions[result["ts"]] = {
                "outlook_event_id": outlook_event_id,
                "invite_subject": invite_subject,
                "created_at": datetime.now().isoformat(),
            }

        return result

    def verify_slack_signature(self, timestamp: str, body: str, signature: str) -> bool:
        """Verify a Slack request signature (for webhook security)."""
        if not self._config.slack_signing_secret:
            return False
        basestring = f"v0:{timestamp}:{body}"
        my_sig = "v0=" + hmac.new(
            self._config.slack_signing_secret.encode(),
            basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(my_sig, signature)

    async def handle_interaction(self, payload: dict) -> dict:
        """Process a Slack interactive response (button click).

        This closes the "Outlook-Slack Loop":
        1. User clicks Accept/Decline in Slack
        2. This method routes to OutlookManager.respond_to_invite
        3. Sends confirmation back to Slack
        """
        actions = payload.get("actions", [])
        if not actions:
            return {"status": "no_action"}

        action = actions[0]
        action_id = action.get("action_id", "")
        event_id = action.get("value", "")
        user = payload.get("user", {}).get("name", "someone")
        channel = payload.get("channel", {}).get("id", self._config.slack_channel)

        if action_id in ("accept_invite", "decline_invite"):
            verb = "accept" if action_id == "accept_invite" else "decline"

            # Route to Outlook
            result = await mcp_bridge.outlook.respond_to_invite(event_id, verb)

            # Confirm in Slack
            status_emoji = "✅" if verb == "accept" else "❌"
            await self.send_message(
                text=f"{status_emoji} {user} {verb}ed the invite via NEXUS.",
                channel=channel,
            )

            logger.info(f"MCP Loop closed: {user} {verb}ed invite {event_id}")
            return {"status": "ok", "action": verb, "event_id": event_id}

        return {"status": "unknown_action", "action_id": action_id}


# ═══════════════════════════════════════════════════════════════
# MCP BRIDGE — Unified entry point
# ═══════════════════════════════════════════════════════════════

class MCPBridge:
    """The unified MCP Executive Bridge.

    Combines OutlookManager + SlackMessenger into a single interface
    that the NEXUS agent system can call as MCP tools.

    Zero-Token Fail-Safe: If credentials are missing, all methods
    return structured "disabled" responses. The core NEXUS system
    is NEVER affected.
    """

    def __init__(self):
        self.config = load_mcp_config()
        self.outlook = OutlookManager(self.config)
        self.slack = SlackMessenger(self.config)

        # Log status
        services = []
        if self.config.outlook_enabled:
            services.append("Outlook")
        if self.config.slack_enabled:
            services.append("Slack")

        if services:
            logger.info(f"MCP Executive Bridge initialized: {', '.join(services)} active")
        else:
            logger.info(
                "MCP Executive Bridge initialized in DISABLED mode. "
                "Set MICROSOFT_GRAPH_CLIENT_ID, MICROSOFT_GRAPH_CLIENT_SECRET, "
                "SLACK_BOT_TOKEN to enable enterprise integrations."
            )

    @property
    def enabled(self) -> bool:
        return self.config.outlook_enabled or self.config.slack_enabled

    @property
    def fully_enabled(self) -> bool:
        return self.config.fully_enabled

    async def process_outlook_invite(self, invite_subject: str, invite_start: str,
                                     invite_end: str, outlook_event_id: str = "") -> dict:
        """The complete "Outlook-Slack Loop" workflow:

        1. Detect conflicts (Outlook + NEXUS calendars)
        2. Generate conflict report
        3. Send interactive Slack notification
        4. (User clicks Accept/Decline in Slack)
        5. (handle_interaction closes the loop)

        Returns the conflict report and Slack notification status.
        """
        # Step 1: Detect conflicts
        conflicts = await self.outlook.detect_conflicts(invite_start, invite_end)

        # Step 2 + 3: Send to Slack
        slack_result = await self.slack.send_conflict_report(
            invite_subject=invite_subject,
            invite_start=invite_start,
            invite_end=invite_end,
            conflicts=conflicts,
            outlook_event_id=outlook_event_id,
        )

        return {
            "invite": {
                "subject": invite_subject,
                "start": invite_start,
                "end": invite_end,
            },
            "conflicts": conflicts,
            "slack_notification": slack_result,
            "status": "awaiting_response" if slack_result.get("status") == "ok" else slack_result.get("status"),
        }

    def get_status(self) -> dict:
        """Get MCP Bridge health status."""
        return {
            "outlook": {
                "enabled": self.config.outlook_enabled,
                "authenticated": bool(self.config.ms_access_token),
                "token_valid": time.time() < self.config.ms_token_expiry if self.config.ms_token_expiry else False,
            },
            "slack": {
                "enabled": self.config.slack_enabled,
                "channel": self.config.slack_channel if self.config.slack_enabled else None,
            },
            "fully_enabled": self.config.fully_enabled,
            "pending_actions": len(self.slack._pending_actions),
        }


# ── Module-level singleton — safe even without credentials ────
try:
    mcp_bridge = MCPBridge()
except Exception as e:
    logger.error(f"MCP Bridge failed to initialize: {e}")
    mcp_bridge = None
