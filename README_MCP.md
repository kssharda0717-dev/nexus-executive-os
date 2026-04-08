# NEXUS MCP Executive Bridge

## What is this?

The **MCP Executive Bridge** connects NEXUS's local autonomous agent mesh to enterprise systems using the **Model Context Protocol** pattern. It bridges siloed data (Outlook email/calendar) with communication layers (Slack), enabling NEXUS to operate as a true life operating system — not just a local task manager.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NEXUS Agent Mesh                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Planning  │  │ Learning │  │LifeAdmin │  │ Proactive  │ │
│  │  Agent    │  │  Agent   │  │  Agent   │  │  Monitor   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘ │
│       └──────────────┴──────────────┴──────────────┘        │
│                          │                                   │
│                ┌─────────┴─────────┐                        │
│                │  Primary Agent    │                        │
│                │  (Orchestrator)   │                        │
│                └─────────┬─────────┘                        │
└──────────────────────────┼──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │   MCP Executive Bridge  │
              │  (mcp_executive.py)     │
              ├─────────┬───────────────┤
              │         │               │
     ┌────────┴──┐  ┌───┴────────┐     │
     │  Outlook  │  │   Slack    │     │
     │  Manager  │  │ Messenger  │     │
     │           │  │            │     │
     │ • Emails  │  │ • Messages │     │
     │ • Calendar│  │ • Blocks   │     │
     │ • Invites │  │ • Actions  │     │
     └─────┬─────┘  └─────┬─────┘     │
           │               │           │
   ┌───────┴───┐   ┌──────┴──────┐   │
   │ Microsoft │   │   Slack     │   │
   │ Graph API │   │   Web API   │   │
   └───────────┘   └─────────────┘   │
              └────────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │  "Outlook-Slack Loop"   │
              │                         │
              │  1. Outlook invite in   │
              │  2. Conflict detection  │
              │  3. Slack notification  │
              │  4. User clicks button  │
              │  5. Action executed     │
              └─────────────────────────┘
```

## The "Outlook-Slack Loop"

This is the flagship MCP workflow:

1. **Trigger**: An Outlook calendar invite is detected
2. **Conflict Detection**: NEXUS checks both the Outlook Calendar AND the local NEXUS calendar for time conflicts
3. **Slack Notification**: A formatted Block Kit message is sent to Slack with the invite details, conflict report, and interactive Accept/Decline buttons
4. **User Decision**: The user clicks Accept or Decline in Slack
5. **Execution**: NEXUS routes the action back to Microsoft Graph to accept/decline the invite
6. **Confirmation**: Slack receives a confirmation message

## Zero-Token Fail-Safe

The MCP Bridge is designed to **never crash NEXUS**:

- If `MICROSOFT_GRAPH_CLIENT_ID` is missing → Outlook methods return `{"status": "disabled"}`
- If `SLACK_BOT_TOKEN` is missing → Slack methods return `{"status": "disabled"}`
- If **both** are missing → The bridge runs in fully disabled mode; all API endpoints return structured 501 responses
- The core dashboard, local agents, calendar, tasks, and notes work **perfectly** without any MCP credentials

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/mcp/status` | Bridge health (Outlook + Slack connectivity) |
| GET | `/mcp/outlook/auth` | Start Microsoft OAuth2 flow |
| GET | `/mcp/outlook/callback` | OAuth2 redirect handler |
| GET | `/mcp/outlook/emails` | Fetch unread Outlook emails |
| GET | `/mcp/outlook/calendar` | Fetch Outlook calendar events |
| POST | `/mcp/outlook/check-invite` | Run the full conflict-detection + Slack notification loop |
| POST | `/mcp/slack/interactions` | Slack webhook for button clicks |
| POST | `/mcp/slack/send` | Send a message to Slack |

## Quick Start (3 Steps)

### Step 1: Microsoft Azure — Get Outlook Credentials

1. Go to [Azure Portal](https://portal.azure.com) → **Azure Active Directory** → **App registrations**
2. Click **New registration**:
   - Name: `NEXUS Executive Bridge`
   - Redirect URI: `http://localhost:8000/mcp/outlook/callback` (Web)
3. After creation, copy:
   - **Application (client) ID** → `MICROSOFT_GRAPH_CLIENT_ID`
   - Go to **Certificates & secrets** → New client secret → copy value → `MICROSOFT_GRAPH_CLIENT_SECRET`
4. Go to **API permissions** → Add:
   - `Mail.Read` (Delegated)
   - `Calendars.ReadWrite` (Delegated)
   - `User.Read` (Delegated)
5. Click **Grant admin consent**

### Step 2: Slack — Create Bot and Get Token

1. Go to [Slack API](https://api.slack.com/apps) → **Create New App** → From scratch
   - App Name: `NEXUS`
   - Workspace: Select your workspace
2. Go to **OAuth & Permissions** → Add Bot Token Scopes:
   - `chat:write`
   - `channels:read`
3. **Install to Workspace** → Copy **Bot User OAuth Token** → `SLACK_BOT_TOKEN`
4. Go to **Basic Information** → Copy **Signing Secret** → `SLACK_SIGNING_SECRET`
5. Go to **Interactivity & Shortcuts** → Enable → Request URL:
   - `https://your-ngrok-url/mcp/slack/interactions` (use ngrok for local dev)
6. Invite the bot to your channel: `/invite @NEXUS`

### Step 3: Configure NEXUS

```bash
# Copy template and fill in credentials
cp .env.template .env

# Edit .env with your credentials from Steps 1-2
nano .env

# Start NEXUS
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Authenticate Outlook (open in browser)
open http://localhost:8000/mcp/outlook/auth

# Verify everything works
curl http://localhost:8000/mcp/status
```

## File Structure

```
nexus/
├── mcp_executive.py      # MCP Bridge core (OutlookManager + SlackMessenger)
├── routers/mcp.py        # REST endpoints for MCP tools
├── .env.template          # Credential placeholders
└── README_MCP.md          # This file
```
