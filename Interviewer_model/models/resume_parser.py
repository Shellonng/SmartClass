"""
Resume Parser
简历解析模块 - 提取技能、经验等信息
"""

import pdfplumber
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import docx

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ResumeParser:
    """
    简历解析器：从PDF/DOCX提取结构化信息
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化简历解析器
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        
        # 常见技能关键词库（可扩展）
        self.skill_keywords = self._load_skill_keywords()
        
        logger.info("简历解析器初始化完成")
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析简历文件
        
        Args:
            file_path: 简历文件路径（PDF或DOCX）
            
        Returns:
            解析结果字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"简历文件不存在: {file_path}")
        
        logger.info(f"开始解析简历: {file_path.name}")
        
        # 提取文本
        text = self._extract_text(file_path)
        
        if not text:
            logger.warning("简历文本提取失败")
            return self._empty_result()
        
        # 解析各部分
        basic_info = self._extract_basic_info(text)
        result = {
            'name': self._extract_name(text),
            'contact': self._extract_contact(text),
            'basic_info': basic_info,
            'skills': self._extract_skills(text),
            'education': self._extract_education(text),
            'experience': self._extract_experience(text),
            'projects': self._extract_projects(text),
            'raw_text': text
        }
        
        logger.info(f"简历解析完成: 提取到{len(result['skills'])}个技能")
        
        return result
    
    def _extract_text(self, file_path: Path) -> str:
        """
        从文件提取文本
        
        Args:
            file_path: 文件路径
            
        Returns:
            提取的文本
        """
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.pdf':
                return self._extract_from_pdf(file_path)
            elif suffix in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            else:
                logger.error(f"不支持的文件格式: {suffix}")
                return ""
        except Exception as e:
            logger.error(f"文本提取失败: {str(e)}")
            return ""
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """从PDF提取文本"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """从DOCX提取文本"""
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    def _extract_name(self, text: str) -> str:
        """
        提取姓名（简单规则）
        
        Args:
            text: 简历文本
            
        Returns:
            姓名
        """
        # 查找"姓名："或"Name:"后的内容
        patterns = [
            r'姓\s*名[:：]\s*([^\n]+)',
            r'Name[:：]\s*([^\n]+)',
            r'^([^\n]{2,4})\s*\n',  # 第一行可能是姓名
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # 移除可能的职位信息
                name = re.sub(r'[\(（].*?[\)）]', '', name).strip()
                if 2 <= len(name) <= 10:  # 姓名长度合理
                    return name
        
        return "未知"
    
    def _extract_basic_info(self, text: str) -> Dict[str, Any]:
        """
        提取基本信息
        
        Args:
            text: 简历文本
            
        Returns:
            基本信息字典
        """
        info = {}
        
        # 提取籍贯
        origin_patterns = [
            r'籍\s*贯[:：]\s*([^\n]+)',
            r'户\s*籍[:：]\s*([^\n]+)',
        ]
        for pattern in origin_patterns:
            match = re.search(pattern, text)
            if match:
                info['origin'] = match.group(1).strip()
                break
        
        # 提取出生年月
        birth_patterns = [
            r'出生年?月[:：]\s*([^\n]+)',
            r'生日[:：]\s*([^\n]+)',
            r'出生日期[:：]\s*([^\n]+)',
        ]
        for pattern in birth_patterns:
            match = re.search(pattern, text)
            if match:
                info['birth_date'] = match.group(1).strip()
                break
        
        # 提取政治面貌
        political_patterns = [
            r'政治面貌[:：]\s*([^\n]+)',
        ]
        for pattern in political_patterns:
            match = re.search(pattern, text)
            if match:
                info['political_status'] = match.group(1).strip()
                break
        
        # 提取性别
        gender_patterns = [
            r'性\s*别[:：]\s*([男女])',
        ]
        for pattern in gender_patterns:
            match = re.search(pattern, text)
            if match:
                info['gender'] = match.group(1).strip()
                break
        
        return info
    
    def _extract_contact(self, text: str) -> Dict[str, str]:
        """
        提取联系方式
        
        Args:
            text: 简历文本
            
        Returns:
            联系方式字典
        """
        contact = {}
        
        # 邮箱
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact['email'] = email_match.group(0)
        
        # 电话
        phone_pattern = r'1[3-9]\d{9}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact['phone'] = phone_match.group(0)
        
        return contact
    
    def _extract_skills(self, text: str) -> List[str]:
        """
        提取技能列表
        
        Args:
            text: 简历文本
            
        Returns:
            技能列表
        """
        found_skills = set()
        
        # 方法1: 查找"技能"部分
        skill_section_pattern = r'(?:技能|Skills?|专业技能)[:：\s]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n(?:教育|经历|项目))'
        skill_section_match = re.search(skill_section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if skill_section_match:
            skill_text = skill_section_match.group(1)
            # 从技能部分提取
            for category, keywords in self.skill_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in skill_text.lower():
                        found_skills.add(keyword)
        
        # 方法2: 在全文中搜索关键词
        text_lower = text.lower()
        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_skills.add(keyword)
        
        # 去重并排序
        skills = sorted(list(found_skills))
        
        return skills
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """
        提取教育背景
        
        Args:
            text: 简历文本
            
        Returns:
            教育背景列表
        """
        education = []
        
        # 查找教育部分
        edu_pattern = r'(?:教育背景|教育经历|Education)[:：\s]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n(?:工作|项目|技能|荣誉|科研))'
        edu_match = re.search(edu_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if edu_match:
            edu_text = edu_match.group(1)
            
            # 查找学位
            degree_pattern = r'(本科|硕士|博士|学士|Bachelor|Master|PhD)'
            degree_match = re.search(degree_pattern, edu_text, re.IGNORECASE)
            
            # 查找学校
            school_pattern = r'([^，。\n]{2,15}(?:大学|学院|University|College))'
            school_match = re.search(school_pattern, edu_text)
            
            # 查找专业
            major_patterns = [
                r'(?:专业|Major)[:：]\s*([^\n]+)',
                r'(?:软件学院|计算机学院).*?[:：\s]([^\n]+?)(?:专业|方向)',
            ]
            major = None
            for pattern in major_patterns:
                major_match = re.search(pattern, edu_text, re.IGNORECASE)
                if major_match:
                    major = major_match.group(1).strip()
                    break
            
            # 查找毕业年份
            grad_year_pattern = r'(\d{4})\s*[年届]'
            grad_year_match = re.search(grad_year_pattern, edu_text)
            
            # 查找GPA/成绩
            gpa_patterns = [
                r'GPA[:：\s]*([0-9.]+)',
                r'绩点[:：\s]*([0-9.]+)',
                r'必修.*?平均[:：\s]*([0-9.]+)',
                r'加权.*?排名[:：\s]*([0-9/]+)',
            ]
            gpa_info = []
            for pattern in gpa_patterns:
                gpa_match = re.search(pattern, edu_text, re.IGNORECASE)
                if gpa_match:
                    gpa_info.append(gpa_match.group(1))
            
            edu_item = {}
            if degree_match:
                edu_item['degree'] = degree_match.group(1)
            if school_match:
                edu_item['school'] = school_match.group(1)
            if major:
                edu_item['major'] = major
            if grad_year_match:
                edu_item['graduation_year'] = int(grad_year_match.group(1))
            if gpa_info:
                edu_item['gpa'] = ', '.join(gpa_info)
            
            if edu_item:
                education.append(edu_item)
        
        return education
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """
        提取工作经历
        
        Args:
            text: 简历文本
            
        Returns:
            工作经历列表
        """
        experience = []
        
        # 查找工作经历部分
        exp_pattern = r'(?:工作经历|工作经验|Experience)[:：\s]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n(?:项目|教育|技能)|$)'
        exp_match = re.search(exp_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if exp_match:
            exp_text = exp_match.group(1)
            
            # 分割多个工作经历（按空行或时间段）
            # 简化处理：查找时间段来分割
            time_pattern = r'(\d{4}[./\-年]\d{1,2}|\d{4})\s*[-~至]\s*(\d{4}[./\-年]\d{1,2}|\d{4}|至今|present)'
            time_matches = re.finditer(time_pattern, exp_text, re.IGNORECASE)
            
            for match in time_matches:
                start_date = match.group(1)
                end_date = match.group(2)
                
                # 在时间段附近查找公司和职位
                context_start = max(0, match.start() - 100)
                context_end = min(len(exp_text), match.end() + 200)
                context = exp_text[context_start:context_end]
                
                # 查找公司名称
                company_patterns = [
                    r'([^\n]{3,25}(?:公司|集团|科技|网络|信息|有限|股份|Corporation|Inc|Ltd))',
                    r'公司[:：]\s*([^\n]+)',
                ]
                company = None
                for pattern in company_patterns:
                    company_match = re.search(pattern, context)
                    if company_match:
                        company = company_match.group(1).strip()
                        break
                
                # 查找职位
                position_patterns = [
                    r'职位[:：]\s*([^\n]+)',
                    r'(工程师|开发|架构师|经理|主管|总监|实习生)',
                ]
                position = None
                for pattern in position_patterns:
                    position_match = re.search(pattern, context)
                    if position_match:
                        position = position_match.group(1).strip()
                        break
                
                exp_item = {
                    'start_date': start_date,
                    'end_date': end_date
                }
                if company:
                    exp_item['company'] = company
                if position:
                    exp_item['position'] = position
                
                # 提取描述（时间段后的文本）
                description_start = match.end()
                description_end = min(len(exp_text), description_start + 300)
                description = exp_text[description_start:description_end].strip()
                if description:
                    exp_item['description'] = description[:150]  # 截取前150字
                
                experience.append(exp_item)
        
        return experience
    
    def _extract_projects(self, text: str) -> List[Dict[str, str]]:
        """
        提取项目经历
        
        Args:
            text: 简历文本
            
        Returns:
            项目列表
        """
        projects = []
        
        # 查找项目部分
        proj_pattern = r'(?:项目经历|项目经验|科研经历|Projects?)[:：\s]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n(?:工作|教育|技能|荣誉)|$)'
        proj_match = re.search(proj_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if proj_match:
            proj_text = proj_match.group(1)
            
            # 按"项目"或"◆"/"•"分割
            proj_segments = re.split(r'(?:\n\s*◆|\n\s*•|\n\s*项目\d+)', proj_text)
            
            for segment in proj_segments[:5]:  # 最多提取5个项目
                if len(segment.strip()) < 20:
                    continue
                
                # 提取项目名称（第一行或"项目名称："后）
                name_patterns = [
                    r'项目.*?[名称主题题][:：]\s*([^\n]+)',
                    r'^([^\n]{5,50})',  # 第一行作为项目名称
                ]
                name = None
                for pattern in name_patterns:
                    name_match = re.search(pattern, segment.strip(), re.MULTILINE)
                    if name_match:
                        name = name_match.group(1).strip()
                        # 清理名称中的项目/负责人等标记
                        name = re.sub(r'(项目|负责人|团队|成员)', '', name).strip()
                        if len(name) > 5:
                            break
                
                # 提取技术栈
                tech_patterns = [
                    r'技术栈[:：]\s*([^\n]+)',
                    r'使用.*?[:：]\s*([^\n]+)',
                    r'基于.*?[:：]\s*([^\n]+)',
                ]
                tech_stack = []
                for pattern in tech_patterns:
                    tech_match = re.search(pattern, segment, re.IGNORECASE)
                    if tech_match:
                        tech_text = tech_match.group(1)
                        # 分割技术栈
                        tech_stack = [t.strip() for t in re.split(r'[、，,、]', tech_text) if t.strip()]
                        break
                
                # 提取职责/负责工作
                resp_patterns = [
                    r'(?:负责|职责).*?[:：]\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|$)',
                    r'负责工作[:：]\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|$)',
                ]
                responsibilities = None
                for pattern in resp_patterns:
                    resp_match = re.search(pattern, segment, re.IGNORECASE | re.DOTALL)
                    if resp_match:
                        responsibilities = resp_match.group(1).strip()[:200]  # 截取前200字
                        break
                
                proj_item = {}
                if name:
                    proj_item['name'] = name
                else:
                    # 如果没有名称，使用前50个字符作为描述
                    proj_item['name'] = segment.strip()[:50] + "..."
                
                if tech_stack:
                    proj_item['tech_stack'] = tech_stack
                
                if responsibilities:
                    proj_item['responsibilities'] = responsibilities
                else:
                    # 如果没有专门的职责部分，使用整段文本
                    proj_item['description'] = segment.strip()[:200]
                
                if proj_item:
                    projects.append(proj_item)
        
        return projects
    
    def _load_skill_keywords(self) -> Dict[str, List[str]]:
        """
        加载技能关键词库
        
        Returns:
            技能分类字典
        """
        return {
            'programming': [
                'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Go', 'Rust',
                'TypeScript', 'PHP', 'Ruby', 'Swift', 'Kotlin'
            ],
            'web_frontend': [
                'React', 'Vue', 'Angular', 'HTML', 'CSS', 'jQuery',
                'Bootstrap', 'Tailwind', 'Webpack', 'Vite'
            ],
            'web_backend': [
                'Django', 'Flask', 'FastAPI', 'Spring', 'Express',
                'Node.js', 'Nest.js', 'Rails', 'Laravel'
            ],
            'database': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle',
                'SQL Server', 'Elasticsearch', 'SQLite'
            ],
            'ai_ml': [
                'PyTorch', 'TensorFlow', 'Scikit-learn', 'Keras',
                'Pandas', 'NumPy', 'NLP', 'CV', '机器学习', '深度学习',
                'Transformers', 'BERT', 'GPT'
            ],
            'devops': [
                'Docker', 'Kubernetes', 'Jenkins', 'Git', 'Linux',
                'AWS', 'Azure', 'GCP', 'CI/CD', 'Nginx'
            ],
            'mobile': [
                'Android', 'iOS', 'React Native', 'Flutter', 'Uni-app'
            ],
            'other': [
                'RESTful', 'GraphQL', 'Microservices', 'gRPC',
                'Socket', 'WebSocket', 'OAuth', 'JWT'
            ]
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'name': '未知',
            'contact': {},
            'basic_info': {},
            'skills': [],
            'education': [],
            'experience': [],
            'projects': [],
            'raw_text': ''
        }

