"""Fixed prompt set. Frozen before measurement (see PREREGISTRATION.md).

Four design constraints, each forced by a documented failure mode in prior work:

1. BALANCED BY CLASS (3 x 5 = 15). Every prior study of speculative decoding on this model
   family reports the effect splits sharply by prompt class -- code/structured prompts gain,
   free prose often loses, sometimes with opposite SIGNS in the same run. An unbalanced set
   makes the headline mean an artifact of the class mixture. The primary endpoint is therefore
   the class-stratified mean (mean of per-class means), not the raw mean over prompts.

2. EVERY PROMPT GENERATES LONG. Published community results establish that speculative gain
   scales with generation length and that short generations can cost more than they win
   (overhead dominates). A deliberately short prompt in the set would depress every
   speculative arm for a reason unrelated to the mechanism under study. Each prompt here is
   written to run past the 400-token cap; `bench.py` asserts this and flags any request that
   terminates early as a length confound rather than silently averaging it in.

3. FIVE PROMPTS PER CLASS, NOT THREE. The interval on every effect in this study comes from a
   cluster bootstrap that resamples PROMPTS, because passes of the same prompt are repeated
   measures rather than independent samples. With three clusters in a class the bootstrap can
   only produce a handful of distinct resamples and the resulting interval is decorative. Five
   is the minimum at which the per-class interval carries information.

4. THINKING IS A FACTOR, NOT A CONSTANT. Most published work on this model runs thinking OFF
   everywhere; real agentic traffic runs it ON. `think` is crossed with class so the
   interaction can be measured instead of assumed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    tag: str
    cls: str          # code | prose | reason | chat | zh
    system: str
    user: str
    think: bool = False


PROMPTS: tuple[Prompt, ...] = (
    # ---------------------------------------------------------------- code (structured, low entropy)
    Prompt("code_py_ratelimiter", "code", "You write production Python.",
           "Write a Python class `RateLimiter` implementing a token bucket, thread-safe with "
           "threading.RLock. Provide `allow(cost=1)`, `remaining()`, and `reset()`. Include full "
           "type hints, a docstring on every method, and below the class a `unittest.TestCase` "
           "covering exhaustion, refill over time, and concurrent access from four threads."),
    Prompt("code_rust_mergesort", "code", "You write idiomatic Rust.",
           "Write a Rust module with `pub fn merge_sort<T: Ord + Clone>(v: &[T]) -> Vec<T>`, plus "
           "an in-place variant, plus a `#[cfg(test)] mod tests` covering empty, single, already "
           "sorted, reverse sorted, duplicates, and a 1000-element pseudo-random case. Add doc "
           "comments explaining the complexity of each function."),
    Prompt("code_nginx_cfg", "code", "You produce configuration files with explanations.",
           "Produce a complete nginx reverse-proxy configuration for an API on 127.0.0.1:8080: "
           "TLS with modern ciphers, HTTP->HTTPS redirect, gzip, 30s proxy timeouts, websocket "
           "upgrade, rate limiting at 20 req/s per IP with burst 40, and a /health endpoint "
           "exempt from access logging. After the config, explain each directive block."),

    # ---------------------------------------------------------------- prose (free text, high entropy)
    Prompt("prose_rainbow", "prose", "You teach patiently and at length.",
           "Explain to a curious 10-year-old what makes a rainbow form. Cover why it is curved, "
           "why the colours are always in the same order, why you sometimes see two, and why you "
           "can never reach the end of one. Use everyday comparisons, no bullet points."),
    Prompt("prose_bookshelves", "prose", "You are a thoughtful essayist.",
           "Write five paragraphs on why people who move to a new city often reorganise their "
           "bookshelves within the first month. Consider memory, identity, and the way objects "
           "stand in for routines. Do not use bullet points or headings."),
    Prompt("prose_letter", "prose", "You write warm, specific correspondence.",
           "Write a long letter from a retiring lighthouse keeper to the person taking over the "
           "post. Cover the practical routines, the weather signs worth trusting, the loneliness "
           "and how to sit with it, and one thing they should never do. No bullet points."),

    # ---------------------------------------------------------------- reason (thinking on)
    Prompt("reason_trains", "reason", "You solve problems step by step.",
           "A train leaves Paris at 14:00 travelling 120 km/h toward Berlin. Another leaves Berlin "
           "at 15:00 travelling 80 km/h toward Paris. The track is 1000 km. Where and when do they "
           "meet? Show every step, then verify your answer a second, independent way, then state "
           "what would change if the second train left an hour earlier instead.", think=True),
    Prompt("reason_zebra", "reason", "You solve problems step by step.",
           "Five houses in a row, each a different colour, each owner a different drink. The green "
           "house is immediately right of the ivory house. Milk is drunk in the middle house. "
           "Coffee is drunk in the green house. The red house owner drinks tea. The first house is "
           "yellow. Derive everything that follows, state explicitly what remains "
           "underdetermined, and explain why.", think=True),
    Prompt("reason_scheduling", "reason", "You solve problems step by step.",
           "A workshop has three machines and seven jobs with durations 4, 2, 7, 3, 6, 1, 5 hours. "
           "Any job can run on any machine but cannot be split. Find an assignment minimising the "
           "makespan. Show your reasoning, prove a lower bound, and say whether your answer "
           "attains it.", think=True),

    # ---------------------------------------------------------------- chat (conversational, long form)
    Prompt("chat_desk_robot", "chat", "You are a friendly desk robot with opinions.",
           "Tell me about your day in detail, then ask me three questions about mine and explain "
           "why you chose each one. Be specific and conversational rather than generic."),
    Prompt("chat_music", "chat", "You are a friendly desk robot with opinions.",
           "What kind of music would a desk robot actually enjoy, and why? Give four concrete "
           "examples with reasons, then describe what you would put on for someone who has had a "
           "long, frustrating day at work."),
    Prompt("chat_advice", "chat", "You are a friendly desk robot with opinions.",
           "A friend says they want to learn to cook but keeps giving up after a week. Talk them "
           "through it the way a friend would: what usually goes wrong, what to do differently, "
           "and what the first four weeks should actually look like."),

    # ---------------------------------------------------------------- zh (Traditional Chinese)
    Prompt("zh_self_intro", "zh", "你是桌面機器人，說話自然、具體。",
           "請詳細介紹你自己：你的角色、你最擅長協助的三類任務（每類舉一個具體例子）、你不擅長"
           "什麼，以及使用者該如何跟你合作才最有效率。請用連貫的段落書寫，不要用條列。"),
    Prompt("zh_tea", "zh", "你是耐心的講解者。",
           "請向完全沒有經驗的人解釋台灣烏龍茶的沖泡：器具、水溫、置茶量、各泡的時間差異，以及"
           "為什麼同一批茶葉在不同泡數會有不同風味。請用連貫段落解釋原理，不要只列步驟。"),
    Prompt("zh_debug", "zh", "你是資深工程師，說明清楚且有條理。",
           "一位同事說他的 Python 服務在生產環境每隔幾小時就記憶體用量爬升然後被 OOM kill，但本"
           "機跑一整天都正常。請詳細說明你會如何一步步定位這個問題、每一步要看什麼證據、常見的"
           "根因有哪些，以及各自該怎麼驗證。"),

    # ---- added to reach 5 per class: a 3-cluster class makes the cluster bootstrap meaningless
    Prompt("code_go_worker", "code", "You write production Go.",
           "Write a Go worker pool: `type Pool struct` with `New(n int)`, `Submit(func() error)`, "
           "`Shutdown(ctx context.Context) error` draining in-flight work, and error collection "
           "via channels. Handle panics in workers without killing the pool. Add a table-driven "
           "test covering submit-after-shutdown, panic recovery, and context cancellation."),
    Prompt("code_sql_report", "code", "You write SQL and explain query plans.",
           "Given tables customers(id,name,region), orders(id,customer_id,total,created_at) and "
           "refunds(order_id,amount,created_at), write a PostgreSQL query returning, per region, "
           "the top 3 customers by net revenue in the last 90 days, with order count and refund "
           "rate. Use window functions. Then explain the expected plan and which indexes you "
           "would add and why."),

    Prompt("prose_market", "prose", "You write vivid, grounded description.",
           "Describe a Sunday morning wet market in a mid-sized Taiwanese city from the point of "
           "view of someone who has shopped there for thirty years. Cover the sounds, the order "
           "of the stalls, the negotiations, and what has changed since they started. Six "
           "paragraphs, no bullet points."),
    Prompt("prose_argument", "prose", "You argue carefully and concede fairly.",
           "Make the strongest case that public libraries should not lend e-books through "
           "commercial platforms, then make the strongest case against your own position, then "
           "say honestly which you find more persuasive and why. Continuous prose, no headings."),

    Prompt("reason_probability", "reason", "You solve problems step by step.",
           "A factory has three machines producing 20%, 30% and 50% of output, with defect rates "
           "5%, 3% and 1%. An item is found defective. What is the probability it came from each "
           "machine? Show Bayes' theorem applied explicitly, then sanity-check that your three "
           "answers sum to one, then explain intuitively why the answer is not simply "
           "proportional to output share.", think=True),
    Prompt("reason_invariant", "reason", "You solve problems step by step.",
           "Nine coins sit on a table, some heads up. A move flips exactly two coins. Determine "
           "for which starting configurations it is possible to reach all-heads. Identify the "
           "invariant, prove it is preserved by every move, and state the complete condition.",
           think=True),

    Prompt("chat_travel", "chat", "You are a friendly desk robot with opinions.",
           "I have four days in a city I have never visited and I hate crowds and museums. Talk "
           "me through how you would plan those four days in general, what you would deliberately "
           "skip, and how to find the good parts of a place without a guidebook. Be specific."),
    Prompt("chat_disagree", "chat", "You are a friendly desk robot with opinions.",
           "Someone tells you that reading fiction is a waste of time compared with reading "
           "non-fiction. Disagree with them properly: take their point seriously first, then "
           "give your reasons, then say what part of their argument you think actually holds."),

    Prompt("zh_moving", "zh", "你是實務經驗豐富的顧問。",
           "一位朋友要從台北搬到台南工作，租屋、交通、生活成本、社交圈都要重新建立。請詳細說明"
           "你會建議他在搬家前三個月、搬家當月、以及搬家後三個月分別處理哪些事，每個階段說明"
           "原因與常見錯誤。請用連貫段落書寫。"),
    Prompt("zh_review", "zh", "你是嚴謹的技術評審。",
           "有人提出要把公司內部一個每天處理三百萬筆交易的批次系統，從每晚跑一次的排程改成即時"
           "串流處理。請詳細分析這個提案：可能的好處、被低估的成本與風險、需要先釐清哪些前提、"
           "以及你會建議用什麼方式分階段驗證。請用連貫段落論述。"),
)

CLASSES: tuple[str, ...] = ("code", "prose", "reason", "chat", "zh")

# Every prompt is written to exceed this; bench.py flags any request that stops early.
MAX_TOKENS: int = 400


def by_class(cls: str) -> tuple[Prompt, ...]:
    return tuple(p for p in PROMPTS if p.cls == cls)


def _validate() -> None:
    counts = {c: len(by_class(c)) for c in CLASSES}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"prompt classes are not balanced: {counts}")
    if len({p.tag for p in PROMPTS}) != len(PROMPTS):
        raise AssertionError("duplicate prompt tags")
    unknown = {p.cls for p in PROMPTS} - set(CLASSES)
    if unknown:
        raise AssertionError(f"unknown classes: {unknown}")


_validate()
