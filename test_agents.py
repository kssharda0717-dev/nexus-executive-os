"""End-to-end agent pipeline test.

Tests the full flow: intent classification → plan generation → sub-agent routing → tool execution.
Requires GEMINI_API_KEY in .env or environment.

Usage:
    python test_agents.py                    # Run all tests
    python test_agents.py simple             # Test simple query (GeneralAgent)
    python test_agents.py planning           # Test "Plan my week" (PlanningAgent)
    python test_agents.py learning           # Test learning curriculum (LearningAgent)
    python test_agents.py life_admin         # Test life event (LifeAdminAgent)
    python test_agents.py proactive          # Test daily briefing (ProactiveMonitor)
"""

import asyncio
import sys
import json
from datetime import datetime

# Must run from nexus directory
import database as db
from mission_control import mission_control


async def seed_data():
    """Seed realistic test data."""
    await db.get_db()

    # Tasks
    await db.create_task("Review Q2 budget proposal", priority="high",
                         due_date="2026-04-03T17:00:00", estimated_minutes=60,
                         tags=["work", "finance"])
    await db.create_task("Prepare hackathon demo", priority="urgent",
                         due_date="2026-04-05T12:00:00", estimated_minutes=180,
                         tags=["hackathon", "demo"])
    await db.create_task("Grocery shopping", priority="medium",
                         due_date="2026-04-02T18:00:00", estimated_minutes=45,
                         tags=["personal", "errands"])
    await db.create_task("Write unit tests for NEXUS", priority="high",
                         due_date="2026-04-04T17:00:00", estimated_minutes=120,
                         tags=["hackathon", "testing"])

    # Events
    await db.create_event("Team standup", "2026-04-02T09:00:00", "2026-04-02T09:30:00",
                          "Daily sync")
    await db.create_event("Lunch with Sarah", "2026-04-02T12:00:00", "2026-04-02T13:00:00")

    # Notes
    await db.create_note("Q2 Goals",
                         "## Q2 Objectives\n- Ship NEXUS v1\n- Grow user base to 1000\n- Hire 2 engineers",
                         tags=["goal", "q2"])

    print("Test data seeded: 4 tasks, 2 events, 1 note\n")


async def test_simple():
    """Test a simple query handled by GeneralAgent."""
    print("=" * 60)
    print("TEST: Simple query → GeneralAgent")
    print("=" * 60)
    from agents.primary import process_message

    run = await db.create_agent_run("Create a task: Buy birthday gift for Mom, due April 10, high priority")
    result = await process_message(run["id"], "Create a task: Buy birthday gift for Mom, due April 10, high priority")

    print(f"\nResponse: {result['response'][:500]}")
    print(f"Tools used: {result['tools_used']}")
    print()


async def test_planning():
    """Test the Planning Agent with 'Plan my week'."""
    print("=" * 60)
    print("TEST: Plan my week → PlanningAgent")
    print("=" * 60)
    from agents.primary import process_message

    run = await db.create_agent_run("Plan my week. Schedule all my open tasks into available calendar slots.")
    result = await process_message(run["id"],
                                   "Plan my week. Schedule all my open tasks into available calendar slots.")

    print(f"\nResponse: {result['response'][:800]}")
    print(f"Tools used: {result['tools_used']}")
    print()


async def test_learning():
    """Test the Learning Agent."""
    print("=" * 60)
    print("TEST: Learning goal → LearningAgent")
    print("=" * 60)
    from agents.primary import process_message

    run = await db.create_agent_run("I want to learn Kubernetes in 2 weeks")
    result = await process_message(run["id"], "I want to learn Kubernetes in 2 weeks")

    print(f"\nResponse: {result['response'][:800]}")
    print(f"Tools used: {result['tools_used']}")
    print()


async def test_life_admin():
    """Test the Life Admin Agent."""
    print("=" * 60)
    print("TEST: Life event → LifeAdminAgent")
    print("=" * 60)
    from agents.primary import process_message

    run = await db.create_agent_run("I'm moving to a new apartment on April 20th")
    result = await process_message(run["id"], "I'm moving to a new apartment on April 20th")

    print(f"\nResponse: {result['response'][:800]}")
    print(f"Tools used: {result['tools_used']}")
    print()


async def test_proactive():
    """Test the Proactive Monitor."""
    print("=" * 60)
    print("TEST: Daily briefing → ProactiveMonitor")
    print("=" * 60)
    from agents.proactive import run_daily_briefing

    result = await run_daily_briefing()

    print(f"\nResponse: {result['response'][:800]}")
    print(f"Tools used: {result.get('tools_used', [])}")
    print()


async def main():
    import os
    os.unlink("nexus.db") if os.path.exists("nexus.db") else None

    await seed_data()

    tests = {
        "simple": test_simple,
        "planning": test_planning,
        "learning": test_learning,
        "life_admin": test_life_admin,
        "proactive": test_proactive,
    }

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name in tests:
            await tests[test_name]()
        else:
            print(f"Unknown test: {test_name}. Available: {list(tests.keys())}")
    else:
        for name, test_fn in tests.items():
            try:
                await test_fn()
            except Exception as e:
                print(f"FAILED: {name} — {e}\n")
            # Stagger between tests to avoid rate limit bursts
            await asyncio.sleep(3.0)

    # Print final state
    print("=" * 60)
    print("FINAL SYSTEM STATE")
    print("=" * 60)
    tasks = await db.list_tasks()
    print(f"\nTasks ({len(tasks)}):")
    for t in tasks:
        print(f"  [{t['status']}] {t['title']} (priority={t['priority']}, due={t.get('due_date', 'none')})")

    events = await db.list_events()
    print(f"\nEvents ({len(events)}):")
    for e in events:
        print(f"  {e['title']} ({e['start_time']} → {e['end_time']})")

    notes = await db.list_notes()
    print(f"\nNotes ({len(notes)}):")
    for n in notes:
        print(f"  {n['title']} (tags={n.get('tags', [])})")

    await db.close_db()


if __name__ == "__main__":
    asyncio.run(main())
