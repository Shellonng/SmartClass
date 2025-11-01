"""
Simple RAG (Retrieval-Augmented Generation)
简单的RAG检索模块 - 无需数据库知识
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleRAG:
    """
    简单RAG实现：基于向量相似度的问题检索
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RAG检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.rag_config = config.get('rag', {})
        self.embedding_config = config['models']['embedding']
        
        # 加载嵌入模型
        logger.info(f"加载嵌入模型: {self.embedding_config['name']}")
        self.embed_model = SentenceTransformer(self.embedding_config['name'])
        
        # 参数
        self.top_k = self.rag_config.get('top_k', 5)
        self.similarity_threshold = self.rag_config.get('similarity_threshold', 0.6)
        
        # 加载问题库
        self.questions_db: List[Dict[str, Any]] = []
        self.question_embeddings: Optional[np.ndarray] = None
        
        # 尝试加载预存的问题库
        db_path = self.rag_config.get('question_db_path', 'data/interview_questions.json')
        if Path(db_path).exists():
            self.load_question_db(db_path)
        else:
            logger.warning(f"问题库文件不存在: {db_path}")
        
        logger.info("RAG检索器初始化完成")
    
    def load_question_db(self, db_path: str):
        """
        从JSON文件加载问题库
        
        Args:
            db_path: 问题库文件路径
        """
        logger.info(f"加载问题库: {db_path}")
        
        with open(db_path, 'r', encoding='utf-8') as f:
            self.questions_db = json.load(f)
        
        logger.info(f"问题库加载完成: {len(self.questions_db)}条问题")
        
        # 计算所有问题的嵌入向量
        self._compute_embeddings()
    
    def _compute_embeddings(self):
        """
        计算问题库中所有问题的嵌入向量
        """
        if not self.questions_db:
            logger.warning("问题库为空，跳过嵌入计算")
            return
        
        logger.info("开始计算问题嵌入向量...")
        
        questions_text = [q['question'] for q in self.questions_db]
        self.question_embeddings = self.embed_model.encode(
            questions_text,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"嵌入向量计算完成: shape={self.question_embeddings.shape}")
    
    def search(
        self,
        query: str,
        job_title: Optional[str] = None,
        skill: Optional[str] = None,
        difficulty: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关问题
        
        Args:
            query: 查询文本（可以是技能、话题或问题描述）
            job_title: 岗位筛选
            skill: 技能筛选
            difficulty: 难度筛选 (basic/intermediate/advanced)
            top_k: 返回top-k结果
            
        Returns:
            相关问题列表
        """
        if self.question_embeddings is None:
            logger.warning("问题库未初始化")
            return []
        
        top_k = top_k or self.top_k
        
        # 1. 计算查询向量
        query_vec = self.embed_model.encode([query], convert_to_numpy=True)[0]
        
        # 2. 计算余弦相似度
        # 归一化
        query_norm = query_vec / np.linalg.norm(query_vec)
        embeddings_norm = self.question_embeddings / np.linalg.norm(
            self.question_embeddings, axis=1, keepdims=True
        )
        
        # 点积 = 余弦相似度
        similarities = np.dot(embeddings_norm, query_norm)
        
        # 3. 过滤条件
        filtered_indices = []
        for i, q in enumerate(self.questions_db):
            # 相似度阈值
            if similarities[i] < self.similarity_threshold:
                continue
            
            # 岗位筛选
            if job_title and job_title not in q.get('related_jobs', []):
                continue
            
            # 技能筛选
            if skill and skill not in q.get('skills', []):
                continue
            
            # 难度筛选
            if difficulty and q.get('difficulty') != difficulty:
                continue
            
            filtered_indices.append(i)
        
        # 4. 排序并返回top-k
        if not filtered_indices:
            logger.warning(f"未找到匹配的问题: query={query}")
            return []
        
        filtered_sims = similarities[filtered_indices]
        top_indices_in_filtered = np.argsort(filtered_sims)[-top_k:][::-1]
        top_indices = [filtered_indices[i] for i in top_indices_in_filtered]
        
        # 5. 构建结果
        results = []
        for idx in top_indices:
            result = self.questions_db[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            results.append(result)
        
        logger.info(f"检索完成: query={query}, 返回{len(results)}条结果")
        
        return results
    
    def search_by_resume_skills(
        self,
        resume_skills: List[str],
        job_title: str,
        questions_per_skill: int = 2
    ) -> List[Dict[str, Any]]:
        """
        根据简历技能检索问题
        
        Args:
            resume_skills: 简历中的技能列表
            job_title: 目标岗位
            questions_per_skill: 每个技能返回的问题数
            
        Returns:
            问题列表
        """
        all_questions = []
        seen_questions = set()
        
        for skill in resume_skills:
            questions = self.search(
                query=skill,
                job_title=job_title,
                top_k=questions_per_skill * 2  # 多检索一些，去重后保证数量
            )
            
            for q in questions:
                q_id = q.get('id') or q['question']
                if q_id not in seen_questions:
                    all_questions.append(q)
                    seen_questions.add(q_id)
                    
                    if len([qq for qq in all_questions if skill in qq.get('skills', [])]) >= questions_per_skill:
                        break
        
        logger.info(f"为{len(resume_skills)}个技能检索到{len(all_questions)}个问题")
        
        return all_questions
    
    def get_job_core_questions(
        self,
        job_title: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        获取某岗位的核心问题
        
        Args:
            job_title: 岗位名称
            count: 问题数量
            
        Returns:
            核心问题列表
        """
        core_questions = []
        
        for q in self.questions_db:
            if job_title in q.get('related_jobs', []) and q.get('is_core', False):
                core_questions.append(q)
        
        # 按难度分布
        core_questions.sort(key=lambda x: {
            'basic': 1,
            'intermediate': 2,
            'advanced': 3
        }.get(x.get('difficulty', 'intermediate'), 2))
        
        return core_questions[:count]
    
    def add_question(self, question_dict: Dict[str, Any]):
        """
        动态添加新问题
        
        Args:
            question_dict: 问题字典
        """
        self.questions_db.append(question_dict)
        
        # 更新嵌入向量
        if self.question_embeddings is not None:
            new_emb = self.embed_model.encode(
                [question_dict['question']],
                convert_to_numpy=True
            )
            self.question_embeddings = np.vstack([self.question_embeddings, new_emb])
        
        logger.info(f"添加新问题: {question_dict['question']}")
    
    def save_question_db(self, save_path: str):
        """
        保存问题库到JSON文件
        
        Args:
            save_path: 保存路径
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.questions_db, f, ensure_ascii=False, indent=2)
        
        logger.info(f"问题库已保存: {save_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取问题库统计信息
        
        Returns:
            统计信息字典
        """
        if not self.questions_db:
            return {"total": 0}
        
        stats = {
            "total": len(self.questions_db),
            "by_difficulty": {},
            "by_job": {},
            "by_skill": {}
        }
        
        for q in self.questions_db:
            # 难度统计
            difficulty = q.get('difficulty', 'unknown')
            stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1
            
            # 岗位统计
            for job in q.get('related_jobs', []):
                stats['by_job'][job] = stats['by_job'].get(job, 0) + 1
            
            # 技能统计
            for skill in q.get('skills', []):
                stats['by_skill'][skill] = stats['by_skill'].get(skill, 0) + 1
        
        return stats

