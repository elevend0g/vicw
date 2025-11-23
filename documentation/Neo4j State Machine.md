VICW: The Loop Problem & The Graph Solution

1. The Problem: "The Hydro-Plant Loop"

During our recent 130-turn narrative experiment, the Virtual Infinite Context Window (VICW) worked perfectly for the first 80 turns. However, around turn 110, the story stopped moving forward.

What happened:
The characters got stuck in a "Groundhog Day" scenario. Every few minutes, they would stand at the window, look at the dawn, and propose the exact same plan: "We need to go to the Hydro-Plant."

Why it happened:
Our current system relies on a Rolling Text Summary. It summarizes the past into a paragraph like: "The team is at the substation planning to go to the Hydro-Plant."

Because the context window "slides" forward, it eventually forgot that the characters had already agreed to the plan. The LLM saw "planning" in the summary and "planning" in the recent chat logs, so it logically predicted that the next sentence should be... more planning.

It lacked a way to know: "We have already finished planning. Now we must move."

2. The Proposed Solution: The "State Machine"

To fix this, we need to stop treating the story as just a stream of text and start treating it as a World with Rules. We will upgrade the architecture from a "Text Predictor" to a "State Machine."

The Fix in Simple Terms:
We will give the AI a "Checklist" that lives outside the chat window.

How it works (The New Workflow):

The Listener (Graph Worker):

As the characters speak, a background AI listens.

When Alice says, "Let's go to the Hydro-Plant," the Listener updates the Checklist: "Goal: Go to Hydro-Plant (Status: IN PROGRESS)."

When they arrive, the Listener updates the Checklist: "Goal: Go to Hydro-Plant (Status: COMPLETED)."

The Enforcer (Context Injection):

Before the main AI writes the next scene, we secretly hand it the Checklist.

We tell it: "You are at the Hydro-Plant. The goal 'Go to Hydro-Plant' is checked off. DO NOT suggest going there again."

The Result:

If the AI tries to loop back and suggest the plan again, the "Checklist" prevents it. The story is forced to move forward to the next unchecked box.

3. Technology Stack Changes

Redis: Continues to handle the fast, short-term conversation memory.

Neo4j (New): Acts as the "Checklist" database. It stores the facts of the world (Location, Inventory, Goals) so they never get lost or repeated.