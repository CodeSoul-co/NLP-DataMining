"""
Module for cleaning text data using NLP techniques.
"""
import re
import string
import os
import jieba


class TextCleaner:
    """
    Class for cleaning text data using various NLP techniques.
    """
    
    def __init__(self, language='english'):
        """
        Initialize the TextCleaner class.
        
        Args:
            language (str): Language for NLP processing
        """
        self.language = language
        
        # Initialize stopwords
        if language == 'chinese':
            # Initialize Chinese stopwords
            self.stop_words = self._load_chinese_stopwords()
            # Initialize jieba for Chinese segmentation
            jieba.initialize()
        else:
            # Use basic English stopwords
            self.stop_words = self._load_english_stopwords()
    
    def _load_english_stopwords(self):
        """Load basic English stopwords."""
        # Common English stopwords
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
            "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
            'wouldn', "wouldn't"
        }
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text."""
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)
    
    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def _load_chinese_stopwords(self):
        """Load Chinese stopwords from file or use default set."""
        chinese_stopwords = set()
        # Try to find Chinese stopwords file
        stopwords_path = os.path.join(os.path.dirname(__file__), 'chinese_stopwords.txt')
        
        # If file doesn't exist, use a minimal set of common Chinese stopwords
        if not os.path.exists(stopwords_path):
            chinese_stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                               '或', '一个', '没有', '我们', '你们', '他们', '她们', '它们',
                               '这个', '那个', '这些', '那些', '不', '在', '有', '我', '你',
                               '他', '她', '它', '这', '那', '哪', '什么', '怎么', '如何'}
        else:
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        chinese_stopwords.add(line.strip())
            except Exception as e:
                print(f"Error loading Chinese stopwords: {e}")
                # Fall back to minimal set
                chinese_stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着'}
        
        return chinese_stopwords
    
    def tokenize_text(self, text):
        """Tokenize text based on language."""
        if self.language == 'chinese':
            return list(jieba.cut(text))
        else:
            # Simple word tokenization by splitting on whitespace and punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            return text.split()
    
    def remove_stopwords(self, text):
        """Remove stopwords from text."""
        words = self.tokenize_text(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """Apply stemming to text."""
        # Simple stemming implementation (just returns the text as is)
        # For a real implementation, you would use a stemming algorithm
        return text
    
    def lemmatize_text(self, text):
        """Apply lemmatization to text."""
        # Simple lemmatization implementation (just returns the text as is)
        # For a real implementation, you would use a lemmatization algorithm
        return text
    
    def normalize_whitespace(self, text):
        """Normalize whitespace in text."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_numbers(self, text):
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def remove_special_chars(self, text):
        """Remove special characters from text."""
        return re.sub(r'[^\w\s]', '', text)
    
    def named_entity_recognition(self, text):
        """
        Perform Named Entity Recognition on text.
        
        Returns:
            dict: Dictionary with text and entities
        """
        # Simple implementation that just returns the text without entities
        # For a real implementation, you would use an NER model
        return {
            'text': text,
            'entities': []
        }
    
    def clean_text(self, text, operations=None):
        """
        Clean text using specified operations.
        
        Args:
            text (str): Text to clean
            operations (list, optional): List of cleaning operations to apply.
                                         If None, applies all basic operations.
                                         
        Returns:
            str: Cleaned text
        """
        if operations is None:
            operations = [
                'remove_urls',
                'remove_html_tags',
                'remove_punctuation',
                'remove_stopwords',
                'normalize_whitespace'
            ]
        
        for operation in operations:
            if hasattr(self, operation) and callable(getattr(self, operation)):
                text = getattr(self, operation)(text)
        
        return text
    
    def create_chinese_stopwords_file(self, file_path=None):
        """
        Create a Chinese stopwords file with common stopwords.
        
        Args:
            file_path (str, optional): Path to save the stopwords file.
                                       If None, saves to the default location.
        
        Returns:
            str: Path to the created file
        """
        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), 'chinese_stopwords.txt')
        
        # Common Chinese stopwords
        chinese_stopwords = [
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或', '一个', '没有',
            '我们', '你们', '他们', '她们', '它们', '这个', '那个', '这些', '那些', '不', '在',
            '有', '我', '你', '他', '她', '它', '这', '那', '哪', '什么', '怎么', '如何', '为什么',
            '因为', '所以', '但是', '然而', '如果', '虽然', '尽管', '否则', '此外', '除了', '只是',
            '还是', '还有', '一样', '一直', '一定', '一般', '一种', '一些', '一下', '一切', '一边',
            '一起', '上', '下', '不过', '不要', '不能', '不是', '不会', '只有', '可以', '可能',
            '各', '各种', '各个', '好', '如此', '其', '其他', '其它', '其中', '然后', '所有',
            '那么', '那些', '那样', '那边', '那里', '这么', '这些', '这样', '这边', '这里',
            '过', '跟', '边', '近', '远', '等', '等等', '管', '前', '后', '左', '右', '多', '少',
        ]
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for word in chinese_stopwords:
                    f.write(word + '\n')
            return file_path
        except Exception as e:
            print(f"Error creating Chinese stopwords file: {e}")
            return None
