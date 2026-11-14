"""Skill, education, and certification mappings for resume screening."""

from typing import Dict, List, Set


# ============================================================
# TECHNICAL SKILL GROUPS
# ============================================================
# Each skill has its own canonical name.
# Different spellings / abbreviations are treated as synonyms
# of that specific skill.

SKILL_GROUPS: Dict[str, List[str]] = {

    # --------------------------------------------------------
    # Programming Languages
    # --------------------------------------------------------
    "python": [
        "python",
        "python3",
        "python 3",
        "py",
    ],

    "java": [
        "java",
    ],

    "javascript": [
        "javascript",
        "js",
    ],

    "typescript": [
        "typescript",
        "ts",
    ],

    "c++": [
        "c++",
        "cpp",
    ],

    "c#": [
        "c#",
        "c sharp",
        "csharp",
    ],

    "r": [
        "r programming",
        "r language",
    ],


    # --------------------------------------------------------
    # Data Science / Machine Learning / AI
    # --------------------------------------------------------
    "machine_learning": [
        "machine learning",
        "machine-learning",
        "ml",
    ],

    "deep_learning": [
        "deep learning",
        "deep-learning",
        "dl",
    ],

    "artificial_intelligence": [
        "artificial intelligence",
        "artificial-intelligence",
        "ai",
    ],

    "nlp": [
        "nlp",
        "natural language processing",
        "natural-language processing",
    ],

    "computer_vision": [
        "computer vision",
        "cv",
    ],

    "data_science": [
        "data science",
        "data-science",
    ],

    "data_analysis": [
        "data analysis",
        "data analytics",
    ],

    "statistics": [
        "statistics",
        "statistical analysis",
    ],


    # --------------------------------------------------------
    # Python / Data Libraries
    # --------------------------------------------------------
    "pandas": [
        "pandas",
    ],

    "numpy": [
        "numpy",
        "num py",
    ],

    "scipy": [
        "scipy",
    ],

    "scikit_learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn",
        "scikit_learn",
    ],

    "tensorflow": [
        "tensorflow",
        "tensor flow",
    ],

    "pytorch": [
        "pytorch",
        "py torch",
        "torch",
    ],

    "keras": [
        "keras",
    ],

    "opencv": [
        "opencv",
        "open cv",
    ],


    # --------------------------------------------------------
    # Web Development
    # --------------------------------------------------------
    "django": [
        "django",
    ],

    "flask": [
        "flask",
    ],

    "fastapi": [
        "fastapi",
        "fast api",
    ],

    "react": [
        "react",
        "react.js",
        "reactjs",
    ],

    "nodejs": [
        "node.js",
        "nodejs",
        "node js",
    ],

    "express": [
        "express",
        "express.js",
        "expressjs",
    ],

    "angular": [
        "angular",
        "angular.js",
        "angularjs",
    ],

    "vue": [
        "vue",
        "vue.js",
        "vuejs",
    ],

    "nextjs": [
        "next.js",
        "nextjs",
        "next js",
    ],



    "html5": [
        "html5",
        "html",
    ],

    "css3": [
        "css3",
        "css",
        "cascading style sheets",
    ],

    # --------------------------------------------------------
    # UI / UX Design
    # --------------------------------------------------------
    "figma": [
        "figma",
    ],

    "adobe_xd": [
        "adobe xd",
        "adobe experience design",
    ],

    "wireframing": [
        "wireframing",
        "wireframes",
        "wireframe",
    ],

    "prototyping": [
        "prototyping",
        "prototype",
        "prototypes",
    ],

    "user_research": [
        "user research",
        "user research methods",
    ],

    "ux_design": [
        "ux design",
        "user experience",
        "user experience design",
        "ux",
    ],

    "ui_design": [
        "ui design",
        "user interface",
        "user interface design",
        "ui",
    ],

    "design_systems": [
        "design systems",
        "design system",
    ],

    "usability_testing": [
        "usability testing",
        "usability test",
        "user testing",
    ],

    "responsive_design": [
        "responsive design",
        "responsive web design",
    ],

    "accessibility": [
        "accessibility",
        "web accessibility",
        "accessible design",
    ],

    "information_architecture": [
        "information architecture",
    ],

    # --------------------------------------------------------
    # Databases
    # --------------------------------------------------------
    "sql": [
        "sql",
        "structured query language",
    ],

    "mysql": [
        "mysql",
    ],

    "postgresql": [
        "postgresql",
        "postgres",
        "postgre sql",
    ],

    "oracle": [
        "oracle database",
        "oracle db",
    ],

    "mongodb": [
        "mongodb",
        "mongo db",
        "mongo",
    ],

    "redis": [
        "redis",
    ],

    "cassandra": [
        "cassandra",
    ],

    "elasticsearch": [
        "elasticsearch",
        "elastic search",
    ],

    "nosql": [
        "nosql",
        "no sql",
        "non relational database",
    ],


    # --------------------------------------------------------
    # Cloud Platforms
    # --------------------------------------------------------
    "aws": [
        "aws",
        "amazon web services",
    ],

    "azure": [
        "azure",
        "microsoft azure",
    ],

    "gcp": [
        "gcp",
        "google cloud",
        "google cloud platform",
    ],


    # --------------------------------------------------------
    # AWS Services
    # --------------------------------------------------------
    "ec2": [
        "ec2",
        "amazon ec2",
    ],

    "s3": [
        "s3",
        "amazon s3",
    ],

    "lambda": [
        "lambda",
        "aws lambda",
    ],

    "cloudwatch": [
        "cloudwatch",
        "aws cloudwatch",
    ],


    # --------------------------------------------------------
    # DevOps / Containers
    # --------------------------------------------------------
    "docker": [
        "docker",
        "dockerfile",
        "containerization",
        "containerisation",
    ],

    "kubernetes": [
        "kubernetes",
        "k8s",
    ],

    "helm": [
        "helm",
    ],

    "jenkins": [
        "jenkins",
    ],

    "ci_cd": [
        "ci/cd",
        "ci cd",
        "continuous integration",
        "continuous delivery",
        "continuous deployment",
    ],


    # --------------------------------------------------------
    # Version Control
    # --------------------------------------------------------
    "git": [
        "git",
    ],

    "github": [
        "github",
        "git hub",
    ],

    "gitlab": [
        "gitlab",
        "git lab",
    ],

    "bitbucket": [
        "bitbucket",
    ],


    # --------------------------------------------------------
    # Big Data
    # --------------------------------------------------------
    "spark": [
        "spark",
        "apache spark",
        "pyspark",
        "py spark",
    ],

    "hadoop": [
        "hadoop",
        "apache hadoop",
    ],


    # --------------------------------------------------------
    # NLP / LLM / Generative AI
    # --------------------------------------------------------
    "transformers": [
        "transformers",
        "hugging face transformers",
    ],

    "huggingface": [
        "hugging face",
        "huggingface",
    ],

    "langchain": [
        "langchain",
    ],

    "rag": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
    ],

    "llm": [
        "llm",
        "large language model",
        "large language models",
    ],


    # --------------------------------------------------------
    # API / Backend
    # --------------------------------------------------------
    "rest_api": [
        "rest api",
        "restful api",
        "restful apis",
        "rest apis",
    ],

    "graphql": [
        "graphql",
        "graph ql",
    ],


    # --------------------------------------------------------
    # Project Management / Methodologies
    # --------------------------------------------------------
    "agile": [
        "agile",
    ],

    "scrum": [
        "scrum",
    ],

    "kanban": [
        "kanban",
    ],

    "jira": [
        "jira",
    ],

    # --------------------------------------------------------
    # Cybersecurity / SOC
    # --------------------------------------------------------
    "siem": [
        "siem",
        "security information and event management",
    ],

    "splunk": [
        "splunk",
    ],

    "incident_response": [
        "incident response",
        "incident responses",
        "incident handling",
        "security incident response",
        "security incident handling",
    ],

    "threat_hunting": [
        "threat hunting",
        "threat hunt",
        "threat hunts",
    ],

    "vulnerability_assessment": [
        "vulnerability assessment",
        "vulnerability assessments",
        "vulnerability analysis",
        "vulnerability scanning",
        "vulnerability scans",
    ],

    "firewalls": [
        "firewall",
        "firewalls",
        "firewall configuration",
        "firewall configurations",
    ],

    "network_security": [
        "network security",
        "network-security",
    ],

    "security_monitoring": [
        "security monitoring",
        "security event monitoring",
        "security operations monitoring",
    ],

    "linux": [
        "linux",
        "linux operating system",
    ],

    "tcp_ip": [
        "tcp/ip",
        "tcp ip",
        "tcp-ip",
        "transmission control protocol/internet protocol",
    ],

    "cybersecurity": [
        "cybersecurity",
        "cyber security",
        "cybersecurity fundamentals",
        "cyber security fundamentals",
    ],

    "soc": [
        "soc",
        "security operations center",
        "security operations centre",
    ],

    "ids_ips": [
        "ids/ips",
        "ids ips",
        "ids/ips systems",
        "intrusion detection and prevention",
        "intrusion detection system",
        "intrusion prevention system",
    ],

    "wireshark": [
        "wireshark",
    ],

    "mitre_attack": [
        "mitre att&ck",
        "mitre attack",
        "mitre attck",
    ],


    # --------------------------------------------------------
    # AI / ML Extras (added)
    # --------------------------------------------------------
    "spacy": [
        "spacy",
    ],

    "nltk": [
        "nltk",
    ],

    "matplotlib": [
        "matplotlib",
    ],

    "seaborn": [
        "seaborn",
    ],

    "streamlit": [
        "streamlit",
    ],

    "jupyter": [
        "jupyter",
        "jupyter notebook",
        "jupyter notebooks",
    ],

    "sqlite": [
        "sqlite",
        "sqlite3",
    ],

    # SHAP / Grad-CAM are grouped with explainability on purpose:
    # JDs usually say "SHAP or Grad-CAM" (either is fine), and the
    # matcher cannot express "or", so one shared skill avoids
    # marking a candidate "missing" a tool they did not need.
    "explainable_ai": [
        "explainable ai",
        "explainable artificial intelligence",
        "xai",
        "model explainability",
        "explainability",
        "interpretability",
        "shap",
        "grad-cam",
        "grad cam",
        "gradcam",
    ],

    "generative_ai": [
        "generative ai",
        "generative artificial intelligence",
        "gen ai",
        "genai",
    ],

    "data_visualization": [
        "data visualization",
        "data visualisation",
    ],

    "data_preprocessing": [
        "data preprocessing",
        "data pre-processing",
        "data cleaning",
        "data wrangling",
    ],

    "data_structures_algorithms": [
        "data structures and algorithms",
        "data structures & algorithms",
        "data structures",
        "dsa",
    ],

    "oop": [
        "oop",
        "object oriented programming",
        "object-oriented programming",
    ],

}


# ============================================================
# SKILL CATEGORIES
# ============================================================

SKILL_CATEGORIES: Dict[str, List[str]] = {

    "programming_languages": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "r",
    ],

    "machine_learning_ai": [
        "machine_learning",
        "deep_learning",
        "artificial_intelligence",
        "nlp",
        "computer_vision",
        "data_science",
        "data_analysis",
        "statistics",
    ],

    "ml_libraries": [
        "pandas",
        "numpy",
        "scipy",
        "scikit_learn",
        "tensorflow",
        "pytorch",
        "keras",
        "opencv",
        "spacy",
        "nltk",
        "matplotlib",
        "seaborn",
        "streamlit",
        "jupyter",
    ],

    "web_development": [
        "django",
        "flask",
        "fastapi",
        "react",
        "nodejs",
        "express",
        "angular",
        "vue",
        "nextjs",
    ],

    "ui_ux_design": [
        "figma",
        "adobe_xd",
        "wireframing",
        "prototyping",
        "user_research",
        "ux_design",
        "ui_design",
        "design_systems",
        "usability_testing",
        "responsive_design",
        "accessibility",
        "information_architecture",
    ],

    "databases": [
        "sql",
        "mysql",
        "postgresql",
        "oracle",
        "mongodb",
        "redis",
        "cassandra",
        "elasticsearch",
        "nosql",
        "sqlite",
    ],

    "cloud": [
        "aws",
        "azure",
        "gcp",
        "ec2",
        "s3",
        "lambda",
        "cloudwatch",
    ],

    "devops": [
        "docker",
        "kubernetes",
        "helm",
        "jenkins",
        "ci_cd",
    ],

    "version_control": [
        "git",
        "github",
        "gitlab",
        "bitbucket",
    ],

    "big_data": [
        "spark",
        "hadoop",
    ],

    "generative_ai": [
        "transformers",
        "huggingface",
        "langchain",
        "rag",
        "llm",
    ],

    "backend_api": [
        "rest_api",
        "graphql",
    ],

    "methodologies": [
        "agile",
        "scrum",
        "kanban",
        "jira",
    ],

    "cybersecurity": [
        "siem",
        "splunk",
        "incident_response",
        "threat_hunting",
        "vulnerability_assessment",
        "firewalls",
        "network_security",
        "security_monitoring",
        "linux",
        "tcp_ip",
        "cybersecurity",
        "soc",
        "ids_ips",
        "wireshark",
        "mitre_attack",
    ],

}


# ============================================================
# EDUCATION
# ============================================================

EDUCATION_KEYWORDS: Dict[str, List[str]] = {

    "phd": [
        "phd",
        "ph.d",
        "doctorate",
        "doctoral",
    ],

    "masters": [
        "master's",
        "masters",
        "master",
        "m.s",
        "ms",
        "m.sc",
        "msc",
        "m.tech",
        "mtech",
        "mca",
        "mba",
    ],

    "bachelors": [
        "bachelor's",
        "bachelors",
        "bachelor",
        "b.s",
        "bs",
        "b.sc",
        "bsc",
        "b.tech",
        "btech",
        "b.e",
        "be",
        "bca",
        "b.com",
        "ba",
    ],

    "diploma": [
        "diploma",
    ],
}


# ============================================================
# CERTIFICATIONS
# ============================================================

CERTIFICATION_PATTERNS: List[str] = [

    r"\b(aws|azure|gcp|google cloud)\s+(certified|certification)\b",

    r"\b(certified|certification|certificate|cert)\s+"
    r"(.*?)\s+(professional|associate|expert)\b",

    r"\b(pmp|scrum|agile|cissp|ceh|ccna|ccnp|ocp|oracle)\s+"
    r"(certified|certification)\b",

    r"\b(microsoft|oracle|salesforce|adobe)\s+"
    r"(certified|certification)\b",

]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_keyword(keyword: str) -> str:
    """
    Normalize a skill/keyword for comparison.

    Technical punctuation is intentionally preserved for skills such as
    C++, C#, TCP/IP, MITRE ATT&CK, and scikit-learn.
    """
    keyword = str(keyword).lower().strip()
    keyword = keyword.replace("–", "-").replace("—", "-")
    return " ".join(keyword.split())


# ============================================================
# MAPPING
# ============================================================

def map_keyword_to_group(keyword: str) -> str:
    """
    Convert a skill/synonym into its canonical skill name.

    Example:
        'sklearn' -> 'scikit_learn'
        'py torch' -> 'pytorch'
        'natural language processing' -> 'nlp'
    """

    keyword_norm = normalize_keyword(keyword)

    for group_name, synonyms in SKILL_GROUPS.items():

        normalized_synonyms = {
            normalize_keyword(synonym)
            for synonym in synonyms
        }

        if keyword_norm in normalized_synonyms:
            return group_name

    return keyword_norm


# ============================================================
# SYNONYM EXPANSION
# ============================================================

def expand_keywords_with_synonyms(
    keywords: List[str],
) -> Set[str]:
    """
    Expand detected skills into their synonyms and canonical names.

    Example:
        ['sklearn']

    becomes something similar to:
        {
            'sklearn',
            'scikit-learn',
            'scikit learn',
            'scikit_learn'
        }
    """

    expanded: Set[str] = set()

    for keyword in keywords:

        keyword_norm = normalize_keyword(keyword)

        # Always keep the original keyword.
        expanded.add(keyword_norm)

        # Find canonical skill.
        group = map_keyword_to_group(keyword_norm)

        # If it is a known skill, add its canonical name
        # and all known synonyms.
        if group in SKILL_GROUPS:

            expanded.add(group)

            for synonym in SKILL_GROUPS[group]:
                expanded.add(normalize_keyword(synonym))

    return expanded


# ============================================================
# ACCESSOR FUNCTIONS
# ============================================================

def get_skill_groups() -> Dict[str, List[str]]:
    """
    Return all skill mappings, including technical and UI/UX skills.
    """
    return SKILL_GROUPS.copy()


def get_skill_categories() -> Dict[str, List[str]]:
    """
    Return skill categories.
    """
    return SKILL_CATEGORIES.copy()


def get_education_keywords() -> Dict[str, List[str]]:
    """
    Return education keyword mappings.
    """
    return EDUCATION_KEYWORDS.copy()
