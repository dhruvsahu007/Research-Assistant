import os

class Utils:
    @staticmethod
    def truncate_text(text, max_length=500):
        """Truncate text to max_length with ellipsis if needed"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
        
    @staticmethod
    def get_env_var(key, default=None):
        """Safely get environment variable"""
        return os.environ.get(key, default)
        
    @staticmethod
    def score_source_credibility(source):
        """Simple source credibility scoring"""
        # Basic implementation - can be enhanced with more sophisticated rules
        if 'web' in source.lower():
            if any(domain in source.lower() for domain in ['.edu', '.gov', 'wikipedia']):
                return 0.9  # Higher credibility for educational/government sources
            elif any(domain in source.lower() for domain in ['.org']):
                return 0.7  # Medium credibility for organization sources
            return 0.5  # Default for other web sources
        return 0.8  # Default for document sources
