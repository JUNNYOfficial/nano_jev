"""粗筛词典：中英双语的极性、意图、维度、情绪与主题线索。

**定位**：这里只放「便宜的信号源」。判断逻辑在 :mod:`nano_jev.nj_engine`，本模块
只负责把词条分组摆清楚，便于单独审阅与扩充。

**为什么是子串匹配而不是分词**：中文没有天然词边界，且本子系统刻意保持零依赖、
可在任意机器上单跑。代价是会出现少量误命中（如「好」命中「好久不见」、「卡」命中
「卡片」）——这对**粗筛**可接受，但**不可**当作精选结论，见 ``README.md`` 的局限节。
"""

from __future__ import annotations

# 极性：分「普通」与「强」两档，强词在打分时权重翻倍。
# 每个词条都是**子串**，命中即计一次。

POSITIVE: tuple[str, ...] = (
    "好", "棒", "赞", "强", "厉害", "优秀", "出色", "喜欢", "推荐", "满意",
    "靠谱", "稳", "便宜", "值得", "好用", "流畅", "清晰", "贴心", "给力", "香",
    "准确", "方便", "有用", "顶",
    "good", "great", "nice", "like", "love", "recommend", "satisfied",
    "helpful", "solid", "smooth", "reliable", "useful", "worth",
)

STRONG_POSITIVE: tuple[str, ...] = (
    "惊艳", "牛", "完美", "无敌", "爱了", "太强", "天花板", "封神", "超预期",
    "excellent", "awesome", "amazing", "impressive", "best", "outstanding",
)

NEGATIVE: tuple[str, ...] = (
    "差", "烂", "糟", "垃圾", "失望", "讨厌", "卡", "崩", "慢", "贵", "坑",
    "难用", "后悔", "无语", "恶心", "骗", "报错", "缺", "弱", "凑合", "一般",
    "不稳定", "不支持", "打不开", "闪退",
    "bad", "poor", "slow", "expensive", "buggy", "broken", "unreliable",
    "annoying", "worse", "mediocre", "disappointed",
)

STRONG_NEGATIVE: tuple[str, ...] = (
    "极差", "恶心", "垃圾", "骗人", "骗钱", "坑爹", "退钱", "拉黑", "再也不",
    "terrible", "awful", "hate", "worst", "useless", "scam", "overpriced",
    "crash", "崩了", "没法用",
)

#: 主题词：默认的「相关性」判据领域（AI 模型 / 产品）。可用环境变量或参数覆盖，
#: 使其适配任意评论域。
DEFAULT_TOPICS: tuple[str, ...] = (
    "模型", "大模型", "人工智能", "智能", "产品", "软件", "工具", "应用", "功能",
    "版本", "更新", "上线", "发布", "助手", "对话", "回答", "输出", "提示词",
    "豆包", "通义", "文心", "智谱", "kimi", "qwen", "deepseek", "llama", "gpt",
    "claude", "gemini", "copilot", "chatgpt", "ollama",
    "ai", "llm", "model", "chatbot", "assistant", "api", "prompt", "token",
    "app", "product", "feature", "release", "update",
)

#: 纯噪声（表情/链接/空白）——这些一律判为不相关。
NOISE_MARKERS: tuple[str, ...] = ("http://", "https://", "www.")

#: 意图线索：key 必须落在 ``nj_vocab.INTENTS`` 内；按命中数取多者。
INTENT_CUES: dict[str, tuple[str, ...]] = {
    "question": (
        "？", "?", "吗", "怎么", "如何", "为什么", "为啥", "请问", "求", "有没有",
        "是不是", "能不能", "how", "why", "what", "which",
    ),
    "suggestion": (
        "建议", "希望", "应该", "可以加", "能否", "最好", "期待", "不如", "入手",
        "please", "should", "suggest", "feature request", "would be",
    ),
    "correction": (
        "错误", "不对", "其实", "纠正", "应该是", "搞错", "误", "笔误",
        "wrong", "incorrect", "actually", "correction", "mistake",
    ),
    "praise": (
        "赞", "牛", "厉害", "优秀", "出色", "喜欢", "好", "棒", "顶", "感谢",
        "great", "awesome", "excellent", "love", "thanks", "thank you",
    ),
    "complaint": (
        "差", "烂", "垃圾", "卡", "崩", "慢", "贵", "失望", "坑", "无语", "报错",
        "bug", "broken", "slow", "crash", "terrible", "complaint", "吐槽",
    ),
    "agreement": (
        "同意", "赞同", "说得对", "有道理", "没错", "同感",
        "agreed", "agree", "exactly", "same here", "+1",
    ),
    "disagreement": (
        "不同意", "反对", "胡说", "不见得", "未必", "错",
        "disagree", "not true", "nonsense",
    ),
    "joke": (
        "哈哈", "笑死", "梗", "狗头", "调侃", "玩梗", "lol", "haha", "joke",
    ),
    "information": (
        "分享", "介绍", "说明", "使用", "教程", "实测", "数据", "报告",
        "share", "guide", "fyi", "information", "tested", "benchmark",
    ),
}

#: 维度线索：key 必须落在 ``nj_vocab.ASPECTS`` 内。
ASPECT_CUES: dict[str, tuple[str, ...]] = {
    "capability": (
        "能力", "效果", "智能", "准确", "聪明", "理解", "推理", "回答质量",
        "capability", "smart", "accurate", "reasoning",
    ),
    "quality": ("质量", "品质", "quality"),
    "usability": (
        "易用", "上手", "界面", "交互", "体验", "操作", "设置", "ui", "ux",
        "usability", "interface", "onboarding",
    ),
    "performance": (
        "速度", "快", "慢", "延迟", "响应", "卡顿", "吞吐",
        "speed", "latency", "fast", "slow", "throughput", "performance",
    ),
    "reliability": (
        "稳定", "崩溃", "报错", "掉线", "可用性", "故障", "正常",
        "stability", "stable", "crash", "bug", "reliable", "uptime",
    ),
    "price": (
        "价格", "贵", "便宜", "收费", "免费", "性价比", "订阅", "额度", "计费",
        "price", "cost", "expensive", "cheap", "free", "subscription", "pricing",
    ),
    "service": (
        "客服", "服务", "售后", "支持", "回复", "响应速度",
        "support", "service", "customer",
    ),
    "content": ("内容", "选题", "知识", "资料", "语料", "content", "topic", "coverage"),
    "presentation": (
        "表达", "结构", "排版", "措辞", "逻辑", "语法", "格式",
        "writing", "structure", "formatting", "wording", "grammar",
    ),
    "production": (
        "制作", "画质", "音质", "剪辑", "production", "editing", "render",
    ),
    "trust": (
        "可信", "幻觉", "编造", "引用", "来源", "造谣", "真实",
        "trust", "hallucination", "citation", "factual",
    ),
    "comparison": (
        "对比", "相比", "不如", "优于", "比得上", "平替", "vs", "compare",
        "better than", "worse than", "alternative to",
    ),
}

#: 情绪线索：key 必须落在 ``nj_vocab.EMOTIONS`` 内（中文词即取值）。
EMOTION_CUES: dict[str, tuple[str, ...]] = {
    "认可": ("认可", "赞同", "支持", "喜欢", "点赞", "佩服", "优秀"),
    "惊喜": ("惊艳", "惊喜", "哇", "居然", "没想到", "超出预期", "wow", "surprise"),
    "期待": ("期待", "希望", "盼望", "想用", "等一个", "looking forward", "hope", "wish"),
    "质疑": ("质疑", "怀疑", "真的吗", "未必", "不确定", "存疑", "skeptical", "doubt"),
    "失望": ("失望", "可惜", "遗憾", "心累", "凉了", "disappointed"),
    "担忧": ("担心", "担忧", "怕", "隐患", "风险", "忧虑", "worry", "concern", "risk"),
    "调侃": ("哈哈", "笑死", "狗头", "玩梗", "调侃", "手动", "lol", "haha", "tease"),
    "愤怒": ("愤怒", "气死", "生气", "怒", "滚", "恶心", "angry", "furious", "mad"),
}
