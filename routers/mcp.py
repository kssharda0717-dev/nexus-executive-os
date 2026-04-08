"""MCP Executive Bridge — REST & WebSocket endpoints.

Provides API surface for:
    - Neural Link status (all integrations + tools)
    - Outlook OAuth2 authentication flow
    - Unread email fetching
    - Calendar conflict detection
    - Slack interactive webhook handler
    - Web Search (Tavily) status
    - LinkedIn Networking status

Zero-Token Fail-Safe: All endpoints return structured "disabled" responses
when credentials are missing. They NEVER crash the core NEXUS system.
"""

import json
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

logger = logging.getLogger("nexus.routers.mcp")
router = APIRouter(prefix="/mcp", tags=["mcp-executive"])


def _get_bridge():
    """Lazy-import the MCP bridge to avoid circular imports and handle missing deps."""
    try:
        from mcp_executive import mcp_bridge
        return mcp_bridge
    except Exception as e:
        logger.warning(f"MCP Bridge unavailable: {e}")
        return None


def _get_tool_status(module_path: str) -> dict:
    """Safely get status from a tool module."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return mod.get_status()
    except Exception as e:
        logger.debug(f"Tool status unavailable ({module_path}): {e}")
        return {"name": module_path, "enabled": False}


# ═══════════════════════════════════════════════════════════════
# HEALTH / STATUS — Unified Neural Link endpoint
# ═══════════════════════════════════════════════════════════════

@router.get("/status")
async def mcp_status():
    """Neural Link status — all integrations and tools."""
    bridge = _get_bridge()

    # Base status from MCP Bridge (Outlook + Slack)
    if bridge:
        base_status = bridge.get_status()
    else:
        base_status = {
            "outlook": {"enabled": False},
            "slack": {"enabled": False},
            "fully_enabled": False,
            "pending_actions": 0,
        }

    # Tool statuses
    web_search_status = _get_tool_status("tools.web_search")
    linkedin_status = _get_tool_status("tools.linkedin_agent")

    return {
        "status": "ok" if bridge and bridge.enabled else "partial",
        **base_status,
        "tools": {
            "web_search": web_search_status,
            "linkedin": linkedin_status,
        },
    }


# ═══════════════════════════════════════════════════════════════
# OUTLOOK — OAuth2 + Email + Calendar
# ═══════════════════════════════════════════════════════════════

@router.get("/outlook/auth")
async def outlook_auth():
    """Redirect to Microsoft OAuth2 login page."""
    bridge = _get_bridge()
    if not bridge or not bridge.outlook.enabled:
        return JSONResponse(
            status_code=501,
            content={
                "status": "disabled",
                "message": "Outlook integration not configured. "
                           "Set MICROSOFT_GRAPH_CLIENT_ID and MICROSOFT_GRAPH_CLIENT_SECRET.",
            },
        )
    auth_url = bridge.outlook.get_auth_url()
    return RedirectResponse(url=auth_url)


@router.get("/outlook/callback")
async def outlook_callback(code: str = None, error: str = None):
    """Handle Microsoft OAuth2 callback."""
    if error:
        return JSONResponse(status_code=400, content={"error": error})
    if not code:
        return JSONResponse(status_code=400, content={"error": "No authorization code received"})

    bridge = _get_bridge()
    if not bridge:
        return JSONResponse(status_code=501, content={"error": "MCP Bridge unavailable"})

    result = await bridge.outlook.exchange_code(code)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)

    return {"status": "authenticated", "message": "Outlook connected to NEXUS successfully!"}


@router.get("/outlook/emails")
async def outlook_emails(top: int = 10):
    """Fetch unread emails from Outlook."""
    bridge = _get_bridge()
    if not bridge:
        return {"emails": [], "count": 0, "status": "disabled"}
    return await bridge.outlook.fetch_unread_emails(top=top)


@router.get("/outlook/calendar")
async def outlook_calendar(start: str = None, end: str = None):
    """Fetch Outlook calendar events."""
    bridge = _get_bridge()
    if not bridge:
        return {"events": [], "count": 0, "status": "disabled"}
    return await bridge.outlook.fetch_calendar_events(start=start, end=end)


@router.post("/outlook/check-invite")
async def check_invite(request: Request):
    """Process an Outlook invite through the full "Outlook-Slack Loop".

    Body: {subject, start, end, outlook_event_id?}

    1. Detects conflicts with NEXUS + Outlook calendars
    2. Sends interactive Slack notification
    3. Returns conflict report
    """
    bridge = _get_bridge()
    if not bridge:
        return JSONResponse(
            status_code=501,
            content={"status": "disabled", "message": "MCP Bridge unavailable"},
        )

    body = await request.json()
    subject = body.get("subject", "Unknown Meeting")
    start = body.get("start", "")
    end = body.get("end", "")
    event_id = body.get("outlook_event_id", "")

    if not start or not end:
        return JSONResponse(
            status_code=400,
            content={"error": "start and end are required"},
        )

    result = await bridge.process_outlook_invite(
        invite_subject=subject,
        invite_start=start,
        invite_end=end,
        outlook_event_id=event_id,
    )
    return result


# ═══════════════════════════════════════════════════════════════
# SLACK — Interactive Webhook
# ═══════════════════════════════════════════════════════════════

@router.post("/slack/interactions")
async def slack_interactions(request: Request):
    """Handle Slack interactive component webhooks (button clicks).

    This closes the "Outlook-Slack Loop" — when a user clicks
    Accept/Decline on a conflict report, this endpoint routes
    the action back to OutlookManager.
    """
    bridge = _get_bridge()
    if not bridge:
        return JSONResponse(status_code=501, content={"status": "disabled"})

    # Slack sends form-encoded payload
    form = await request.form()
    payload_str = form.get("payload", "{}")
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid payload"})

    # Optional: Verify Slack signature
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    if bridge.slack._config.slack_signing_secret and timestamp and signature:
        body_bytes = await request.body()
        if not bridge.slack.verify_slack_signature(timestamp, body_bytes.decode(), signature):
            logger.warning("Slack signature verification failed")
            return JSONResponse(status_code=401, content={"error": "Invalid signature"})

    result = await bridge.slack.handle_interaction(payload)
    return result


@router.post("/slack/send")
async def slack_send(request: Request):
    """Send a message to the configured Slack channel.

    Body: {text, channel?, blocks?}
    """
    bridge = _get_bridge()
    if not bridge:
        return {"status": "disabled"}

    body = await request.json()
    return await bridge.slack.send_message(
        text=body.get("text", ""),
        channel=body.get("channel"),
        blocks=body.get("blocks"),
    )
