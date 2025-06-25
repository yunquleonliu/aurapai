from services.knowledge_service import KnowledgeService

# Initialize services as singletons
knowledge_service_instance = KnowledgeService()

def get_knowledge_service() -> KnowledgeService:
    """
    Dependency injector for the KnowledgeService.
    
    Returns:
        KnowledgeService: The singleton instance of the knowledge service.
    """
    return knowledge_service_instance
