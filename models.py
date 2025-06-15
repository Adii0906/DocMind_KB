# enhanced_models.py - Better models for semantic search and QA (under 400MB)
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

class EnhancedLocalModels:
    def __init__(self):
        self.embedding_model = None
        self.qa_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_embedding_model(self):
        """Load better embedding model for semantic search"""
        try:
            # Better embedding models under 400MB
            model_options = [
                "sentence-transformers/all-mpnet-base-v2",    # 420MB - Excellent quality (slightly over but worth it)
                "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # 80MB - Optimized for QA
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 118MB - Good multilingual
                "BAAI/bge-small-en-v1.5"  # 133MB - Very good quality
            ]
            
            # Use the QA-optimized model first
            model_name = model_options[1]  # multi-qa-MiniLM-L6-cos-v1
            print(f"Loading embedding model: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            print(f"✅ Embedding model loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            # Fallback to basic model
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                print("✅ Fallback embedding model loaded")
                return True
            except:
                return False
    
    def load_qa_model(self):
        """Load better QA model for answer generation"""
        try:
            from transformers import pipeline
            
            # Better QA models under 400MB
            qa_model_options = [
                "distilbert-base-cased-distilled-squad",  # 260MB - Excellent for QA
                "deepset/minilm-uncased-squad2",          # 90MB - Good balance
                "microsoft/DialoGPT-medium",              # 350MB - Better conversation
                "google/flan-t5-small"                    # 308MB - Instruction-tuned
            ]
            
            # Try the best QA model first
            model_name = qa_model_options[0]  # distilbert-squad
            print(f"Loading QA model: {model_name}")
            
            self.qa_model = pipeline(
                "question-answering",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=False
            )
            print(f"✅ QA model loaded")
            return True
            
        except Exception as e:
            print(f"❌ Error loading QA model: {e}")
            # Fallback to text generation
            try:
                print("Trying fallback text generation model...")
                self.qa_model = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",
                    device=0 if self.device == "cuda" else -1,
                    max_length=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=50256
                )
                print("✅ Fallback text generation model loaded")
                return True
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return False
    
    def get_embeddings(self, texts):
        """Generate embeddings for texts"""
        if not self.embedding_model:
            if not self.load_embedding_model():
                return None
        
        try:
            # Handle single text vs list
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None
    
    def semantic_search(self, query, texts, top_k=5):
        """Enhanced semantic search with better similarity calculation"""
        if not texts:
            return []
        
        # Get embeddings
        query_embedding = self.get_embeddings([query])
        text_embeddings = self.get_embeddings(texts)
        
        if query_embedding is None or text_embeddings is None:
            return []
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, text_embeddings)[0]
        
        # Get top results with better threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Lower threshold for better recall
                results.append({
                    'text': texts[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return results
    
    def generate_answer(self, context, question):
        """Enhanced answer generation with better QA model"""
        if not self.qa_model:
            if not self.load_qa_model():
                return "❌ QA model not available"
        
        try:
            # Check if it's a QA pipeline or text generation
            if hasattr(self.qa_model, 'task') and self.qa_model.task == 'question-answering':
                # Use QA pipeline (much better for questions)
                result = self.qa_model(
                    question=question,
                    context=context[:2000],  # Limit context length
                    max_answer_len=200,
                    handle_impossible_answer=True
                )
                
                confidence = result.get('score', 0)
                answer = result.get('answer', '').strip()
                
                if confidence > 0.1 and answer:
                    return f"{answer}"
                else:
                    return self._fallback_answer(context, question)
            
            else:
                # Text generation fallback
                return self._generate_text_answer(context, question)
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return self._fallback_answer(context, question)
    
    def _generate_text_answer(self, context, question):
        """Text generation approach"""
        try:
            # Better prompt engineering
            prompt = f"Given the following information, answer the question concisely.\n\nInformation: {context[:800]}\n\nQuestion: {question}\n\nAnswer:"
            
            response = self.qa_model(
                prompt,
                max_length=len(prompt.split()) + 40,
                num_return_sequences=1,
                pad_token_id=50256,
                truncation=True,
                do_sample=True,
                temperature=0.3
            )
            
            # Extract and clean answer
            generated_text = response[0]['generated_text']
            answer = generated_text.replace(prompt, "").strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            if answer and len(answer) > 5:
                return answer
            else:
                return self._fallback_answer(context, question)
                
        except Exception as e:
            return self._fallback_answer(context, question)
    
    def _clean_answer(self, answer):
        """Clean and format the generated answer"""
        # Remove repetitive text
        lines = answer.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in clean_lines:
                clean_lines.append(line)
        
        answer = ' '.join(clean_lines)
        
        # Remove incomplete sentences at the end
        sentences = answer.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 5:
            sentences = sentences[:-1]
        
        answer = '. '.join(sentences)
        if answer and not answer.endswith('.'):
            answer += '.'
            
        return answer[:300]  # Limit length
    
    def _fallback_answer(self, context, question):
        """Fallback answer extraction from context"""
        # Simple keyword matching approach
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Extract key information
        sentences = context.split('.')
        relevant_sentences = []
        
        # Look for sentences containing question keywords
        question_words = [word for word in question_lower.split() 
                         if len(word) > 3 and word not in ['what', 'where', 'when', 'how', 'why', 'which']]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(word in sentence.lower() for word in question_words):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:2]) + '.'
        else:
            # Return first meaningful sentence
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    return sentence + '.'
            
            return "Based on the available information, I couldn't find a specific answer to your question."

# Global instance
enhanced_models = EnhancedLocalModels()

def setup_enhanced_models():
    """Setup function to initialize better models"""
    print("🔄 Setting up enhanced local models...")
    
    embedding_ok = enhanced_models.load_embedding_model()
    qa_ok = enhanced_models.load_qa_model()
    
    if embedding_ok and qa_ok:
        print("✅ All enhanced models ready!")
        return True
    elif embedding_ok:
        print("⚠️ Embedding model ready, QA will use fallback method")
        return True
    else:
        print("❌ Failed to load models")
        return False

def query_enhanced_semantic_search(query, texts, top_k=5):
    """Enhanced semantic search wrapper"""
    return enhanced_models.semantic_search(query, texts, top_k)

def generate_enhanced_answer(context, question):
    """Enhanced answer generation wrapper"""
    return enhanced_models.generate_answer(context, question)

def get_model_info():
    """Get information about loaded models"""
    info = {
        "embedding_model": "multi-qa-MiniLM-L6-cos-v1 (QA-optimized, 80MB)",
        "qa_model": "distilbert-base-cased-distilled-squad (260MB)",
        "total_size": "~340MB",
        "features": [
            "QA-optimized embeddings",
            "BERT-based QA model",
            "Better context understanding",
            "Improved answer extraction"
        ]
    }
    return info