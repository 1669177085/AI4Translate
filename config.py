"""
DeepSeek PDF Translator - Configuration
"""

# DeepSeek API settings
DEEPSEEK_API_KEY = ""     #此处替换为自己的api_key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# Translation settings
MAX_CHARS_PER_BATCH = 4000  # Max characters per API call
MAX_RETRIES = 3             # Max retries on API failure
RETRY_DELAY = 2             # Initial retry delay in seconds
TEMPERATURE = 0.3           # Low temperature for consistent translations

# Summary settings
MAX_CHARS_FOR_SUMMARY = 12000  # Max chars of source text sent for summarization

# PDF output settings
SUMMARY_FONT_SIZE = 11      # Font size for summary page body
SUMMARY_TITLE_FONT_SIZE = 16  # Font size for summary page title
DEFAULT_FONT_SIZE = 10      # Default font size for translated text
MIN_FONT_SIZE = 6           # Minimum font size when squeezing text
LINE_SPACING = 1.2          # Line spacing multiplier

# System prompt for academic paper translation
SYSTEM_PROMPT_TRANSLATE = """\
你是一位专业的学术论文翻译专家。请将以下英文学术论文文本翻译成中文。

翻译要求：
1. 保持学术术语的准确性和一致性，全文术语统一
2. 保留数学公式、变量名、引用标记（如[1]、[2]等）、图表编号（如Fig. 1、Table 2）不翻译
3. 保持原文的学术语气和专业风格，避免口语化表达
4. 对于专有名词和技术术语，首次出现时保留英文原文并用括号附中文翻译，如"Transformer（变换器）"
5. 确保中文表达通顺、符合学术写作规范
6. 注意长句的合理断句，使中文阅读流畅

请直接输出翻译结果，不要添加任何额外说明。"""

# System prompt for paper summarization
SYSTEM_PROMPT_SUMMARIZE = """\
你是一位资深的学术论文审稿人，擅长快速提炼论文的核心贡献。请根据以下论文内容，用中文撰写一份结构化的论文概要。

要求：
1. **论文标题**：翻译论文标题为中文，并保留英文原标题
2. **核心方法**：清晰说明研究采用的方法论、技术路线或理论框架（150-300字）
3. **主要发现**：概括关键实验结果、理论贡献或创新点（150-300字）
4. **结论与意义**：总结研究的结论及其在领域内的影响和意义（100-200字）

注意：
- 使用学术化的中文表达
- 突出论文的创新性和独特贡献
- 对于技术术语，保留英文原词并用括号附中文
- 不要包含任何无关信息或个人评价

请直接输出概要内容，不要添加"概要"或"以下是"之类的开头语。"""
