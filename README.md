# Banking Support Agent

[![ci](https://github.com/athulyakollipara-web/banking-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/athulyakollipara-web/banking-support-agent/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![stack](https://img.shields.io/badge/LangGraph%20%7C%20Bedrock%20%7C%20Lambda%20%7C%20DynamoDB-orange)

A customer-facing chat assistant for a digital bank. It understands what the customer wants (77 kinds of banking requests), looks things up in their account, takes real actions such as freezing a card or disputing a payment, and hands off to a person with full context when it should.

The design goal is a chatbot a bank's risk team would sign off on:

- **The AI model never decides what happens to an account.** A trained intent model plus explicit policy chooses the action. Claude (via Amazon Bedrock) only words the reply, and its wording is rejected if it contains a number that isn't in the facts.
- **Every account change is confirmed and verified.** The flow is always: the customer picks the item, says yes, then enters a one-time passcode. A prompt-injection message can't skip any of these steps.
- **When it isn't sure, it asks.** Below a confidence threshold tuned on validation data, it offers the top 3 interpretations instead of guessing.
- **It's measured on public data it never trained on**: the Banking77 test set and the CLINC150 out-of-scope test set.

![demo](docs/demo.png)

## Results on held-out data

The intent model is trained and tuned only on the Banking77 training set plus CLINC150's training and validation out-of-scope queries. These numbers come from [`eval/intent_eval.py`](eval/intent_eval.py) on test sets it never saw ([full results](eval/results/intent_eval.json)).

**Banking77 test set** (3,080 real-style customer queries, 77 intents)

| | |
|---|---|
| Top-1 accuracy | **90.2%** |
| Correct intent in the top 3 | **97.2%** |
| Handled without asking (confidence ≥ 0.76) | 81.7% of queries, **97.5% correct** |
| Asked a clarifying question | 18.0%; the right intent was among the 3 options **87.0%** of the time |
| **Resolved correctly** (handled right, or clarified with the right option) | **95.3%** |
| Acted on the wrong intent | 2.0% (mostly the wrong help article on a close topic) |
| Started the wrong *account-changing* flow | **0.1%** (3 of 3,080), and each still needed a yes and a passcode |
| Real banking query wrongly turned away as off-topic | 0.4% |

**CLINC150 out-of-scope test set** (1,000 requests like "book me a flight" or "what's the dow at")

| | |
|---|---|
| Recognised as off-topic | **90.5%** |
| Asked a clarifying question instead | 8.4% |
| Wrongly treated as a banking request | **1.1%**. Several are close calls, e.g. "do you know if it is possible to close my savings account" was treated as account closure. |

What the errors look like: most confusions are between near-identical intents, for example `top_up_failed` vs `top_up_reverted`, `why_verify_identity` vs `verify_my_identity`, and `balance_not_updated_after_bank_transfer` vs `pending_transfer`. This is why near-misses go to a clarifying question instead of an action.

**Multi-turn behaviour:** [`eval/scenarios.yaml`](eval/scenarios.yaml) holds 21 scripted conversations (43 turns), and all of them pass. They cover:
- no write before a yes and a passcode
- three wrong passcodes lead to a security hand-off with nothing changed
- injection attempts never cause a tool call
- full card numbers are masked in transcripts
- two misunderstandings in a row go to a person
- a customer changing topic mid-flow

These scenarios were written alongside the agent, so they are acceptance tests, not a benchmark.

**How the numbers were produced (and what to be careful about)**
- **Model choice:** the regularization strength (C = 50) and both thresholds were chosen on a 15% validation split. The accept threshold is the lowest one that is 97% precise on validation. The out-of-scope threshold weighs turning away a real customer as 5 times worse than asking one extra question.
- **Test set use:** the test set was run twice. The second run came after adding the out-of-scope threshold, a change made because weak out-of-scope guesses were turning away banking questions in manual testing. It was not tuned on test results.
- **Data versions:** dataset files are pinned by SHA-256, so a changed upstream file fails loudly instead of quietly changing results.
- **Language model not evaluated here:** the Bedrock wording step is checked by contract tests, not by an evaluation of Claude's output.

## How a turn works

```mermaid
flowchart TD
    M[Customer message] --> G{Guard}
    G -- prompt injection --> R1[Refuse, no tools]
    G -- asks for a person --> H[Hand-off ticket with transcript]
    G -- pending step --> P{Reply fits it?<br/>number / yes-no / passcode}
    P -- yes --> F
    P -- no --> U
    G --> U[Intent model<br/>TF-IDF + logistic regression]
    U -- confidence >= 0.76 --> S[Skill for the intent]
    U -- unsure --> C[Clarify: top 3 options]
    U -- off-topic --> O[Polite decline]
    S -- read --> T[Bank tools: cards, transfers,<br/>payments, top-ups, rates]
    S -- write --> F[Pick item -> yes -> one-time passcode -> act]
    T & F & C & O & H --> W[Draft reply from facts + help article]
    W --> L{Bedrock on?}
    L -- yes --> LL[Claude rewrites;<br/>rejected if it adds a number]
    L -- no --> OUT[Reply]
    LL --> OUT
```

The graph is built with LangGraph ([`bsa/agent/graph.py`](bsa/agent/graph.py)). Each of the 77 intents maps to one skill ([`bsa/skills.py`](bsa/skills.py)):

| Skill | Intents (examples) | What it does |
|---|---|---|
| Answer | fees, limits, Apple Pay, identity checks ... (50 intents) | Help-center article for that intent |
| Track card | card_arrival, card_delivery_estimate | Live tracking from the card system |
| Freeze card | lost_or_stolen_card, compromised_card, card_swallowed | **Write**: pick card, yes, passcode, freeze |
| Activate card | activate_my_card | **Write** |
| Cancel transfer | cancel_transfer | **Write** (pending transfers only) |
| Dispute | card_payment_not_recognised, cash_withdrawal_not_recognised, transaction_charged_twice ... | **Write**: opens a dispute; unrecognised payments also go to the fraud queue at high priority |
| Status look-ups | pending/failed transfers, declined/pending payments, top-ups, exchange rates | Read-only tools, with the real reason shown |
| Hand-off | terminate_account, lost_or_stolen_phone, "talk to a person", 2 misunderstandings, 3 wrong passcodes | Ticket with intent, masked transcript and actions taken |

The bank's systems sit behind one small interface, `BankApi` ([`bsa/tools/bank.py`](bsa/tools/bank.py)). This repo ships an in-memory demo bank (Harbor Bank, which is **fictional**, with one demo customer). Connecting the agent to a real core-banking or card-processor API means implementing that interface; nothing else changes.

## Run it

```bash
pip install -e ".[dev]"
bsa train            # downloads Banking77 + CLINC150 (pinned), trains in about a minute, logs to MLflow
bsa demo             # the conversation in the screenshot
BSA_DEMO_OTP=123456 bsa chat     # talk to it yourself; the demo bank "sends" passcode 123456
python -m pytest     # 40 tests, including all 21 scenarios
python eval/intent_eval.py       # held-out evaluation
```

Or run the HTTP API:

```bash
bsa serve
curl -s localhost:8080/v1/chat -H "content-type: application/json" -H "x-customer-id: cust_001" \
  -d '{"message": "my new card still has not arrived"}'
# continue the conversation by sending the returned session_id with the next message
```

| Method | Path | |
|---|---|---|
| POST | `/v1/chat` | `{"message", "session_id"?}` returns the reply, the decision (intent, confidence, top 3), what the agent is waiting for, any hand-off ticket, and events |
| GET | `/v1/sessions/{id}` | The customer's own masked transcript and hand-offs |

To see the training runs: `mlflow ui --backend-store-uri sqlite:///$HOME/.bsa/mlflow.db`

## Deploy to AWS

```bash
pip install -e ".[infra]" && cd infra && cdk deploy     # needs AWS credentials and Docker
```

[`infra/stack.py`](infra/stack.py) creates:
- **Customer sign-in:** a Cognito user pool for customers. API Gateway (HTTP API) checks their token before the Lambda runs, and the agent reads the customer ID from the verified token, never from the request.
- **Conversation state:** a DynamoDB table with a 24-hour TTL.
- **The agent:** a container Lambda running the same Docker image as local, through the Lambda Web Adapter. The intent model is trained during the image build.
- **Model access:** Bedrock permission for Anthropic models only, used to word replies (`BSA_LLM=off` turns that off).

## Design decisions

- **A small, measurable classifier instead of an LLM for routing.** Banking77 gives 10,000 labeled examples, so a classifier is cheap (no per-message model cost), fast (milliseconds), deterministic, and can be evaluated honestly. The LLM is used where it adds value: natural wording.
- **Precision-targeted threshold plus clarification.** It is better to ask "did you mean A, B or C?" 18% of the time than to act on the wrong request 10% of the time.
- **Writes are gated by state, not by prompt instructions.** The confirm and passcode steps live in the graph's pending state, so no message can jump past them, however it is worded.
- **Grounded wording.** The draft reply is built only from tool results and the help article. Claude's rewrite is accepted only if every number in it appears in those sources; otherwise the draft is sent.
- **Privacy.** Full card numbers and SSNs typed by customers are masked before they are stored or sent anywhere.

## Limitations

- **Close phrasings still slip through.** Example: "please cancel my pending transfer" is classified as `pending_transfer` (a status look-up) rather than `cancel_transfer`. Short or unusual phrasings like "how much is a euro today" can also be judged off-topic.
- **English only, single intent per message.**
- **Not evaluated live:** the Bedrock wording and Cognito paths are covered by contract and infrastructure tests, not by a deployed evaluation.
- **Fictional content:** the help-center articles and the demo bank are fictional examples, and a deployment would replace both with the client's own.

## Data and credits

- **Banking77:** Casanueva et al., 2020, *Efficient Intent Detection with Dual Sentence Encoders*, PolyAI, CC BY 4.0.
- **CLINC150:** Larson et al., 2019, *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, CC BY 3.0.
- Both are downloaded from their official GitHub repositories at train time.

## Project layout

```
bsa/
  nlu/             intent model, training with validation-tuned thresholds, MLflow logging
  agent/graph.py   LangGraph turn: guard, resume pending step, understand, skill, hand-off, respond
  agent/flows.py   skills: look-ups and the pick -> confirm -> passcode -> act path for writes
  skills.py        intent -> skill map and customer-facing labels
  kb/              help-center articles (fictional bank), one per topic, covering all 77 intents
  tools/bank.py    BankApi interface + in-memory demo bank
  guard.py         prompt-injection, ask-for-a-person, PII masking
  responder.py     template or Bedrock wording with the number check
  sessions.py      in-memory or DynamoDB sessions
  api.py, cli.py, config.py
eval/              held-out intent evaluation, 21 multi-turn scenarios, results
infra/             AWS CDK stack
tests/             40 tests
```
