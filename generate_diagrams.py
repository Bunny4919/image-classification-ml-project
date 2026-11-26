import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, color='white', edgecolor='black', fontweight='bold', ha='center'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor=edgecolor, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha=ha, va='center', fontweight=fontweight, fontsize=12, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', va='bottom', fontsize=10)

def create_architecture():
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    plt.title("System Architecture: Gold Jewellery AI", fontsize=20, fontweight='bold', pad=20)
    
    # Components
    # Client
    draw_box(ax, 4, 14, 4, 1, "User Client\n(Frontend: HTML/JS/CSS)", color='#FFF9C4')
    
    # API
    draw_box(ax, 4, 11, 4, 1, "FastAPI Gateway\n(Routing & Orchestration)", color='#F5F5F5')
    
    # Vision
    draw_box(ax, 1, 8, 3, 2, "Vision Module\n- ResNet50 Classifier\n- DINOv2 Embedding", color='#E1F5FE')
    
    # Retrieval
    draw_box(ax, 5, 8, 2, 2, "Vector Search\n(FAISS Index)", color='#E8F5E9')
    
    # RAG
    draw_box(ax, 8, 8, 3, 2, "RAG Engine\n- ChromaDB\n- Gemini LLM", color='#F3E5F5')
    
    # Data/Store
    draw_box(ax, 4, 4, 4, 1, "Jewellery Catalog\n(Stored Metadata & Vectors)", color='#FFF3E0')
    
    # Output
    draw_box(ax, 4, 1, 4, 1, "Final Response\n(Identification & Summary)", color='#FFF9C4')
    
    # Arrows
    draw_arrow(ax, 6, 14, 6, 12, "Upload Image")
    draw_arrow(ax, 4, 11.5, 3, 9.5, "Process Image")
    draw_arrow(ax, 6, 11, 6, 10, "Route Request")
    draw_arrow(ax, 6, 11, 9, 9.5, "Fetch Context")
    
    draw_arrow(ax, 3, 8, 5, 8, "Search Vectors")
    draw_arrow(ax, 7, 8, 8, 8, "Query Metadata")
    
    draw_arrow(ax, 6, 8, 6, 5, "Read/Write")
    
    draw_arrow(ax, 6, 4, 6, 2, "Result Delivery")
    
    plt.tight_layout()
    plt.savefig("system_architecture.jpg", dpi=300, bbox_inches='tight')
    plt.close()

def create_activity():
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    plt.title("UML Activity Diagram: Main Workflow", fontsize=20, fontweight='bold', pad=20)
    
    # Nodes
    # Start
    draw_box(ax, 5.5, 17, 1, 0.5, "Start", color='white')
    
    # Process
    draw_box(ax, 4, 15.5, 4, 0.8, "Upload & Validate Image", color='#F5F5F5')
    draw_box(ax, 4, 14, 4, 0.8, "Center Cropping & Preprocessing", color='#F5F5F5')
    draw_box(ax, 4, 12.5, 4, 0.8, "Multi-view Classification\n(Weighted Probability)", color='#F5F5F5')
    draw_box(ax, 4, 11, 4, 0.8, "DINOv2 Feature Embedding", color='#F5F5F5')
    draw_box(ax, 4, 9.5, 4, 0.8, "Category-Filtered FAISS Search", color='#F5F5F5')
    
    # Decision
    draw_box(ax, 5, 7.5, 2, 1, "Match Found?", color='#FFF9C4')
    
    # Path Yes
    draw_box(ax, 8, 6, 3, 0.8, "Retrieve Metadata\n(ChromaDB)", color='#F5F5F5')
    draw_box(ax, 8, 4.5, 3, 0.8, "Category Verification", color='#F5F5F5')
    draw_box(ax, 8, 3, 3, 0.8, "Generate LLM Summary\n(Gemini)", color='#F5F5F5')
    
    # Path No
    draw_box(ax, 1, 6, 3, 0.8, "Assign Error Status\n(Low Conf/Unknown)", color='#F5F5F5')
    draw_box(ax, 1, 4.5, 3, 0.8, "Generate Explanation", color='#F5F5F5')
    
    # End
    draw_box(ax, 5, 1, 2, 0.5, "End", color='white')
    
    # Arrows
    draw_arrow(ax, 6, 17, 6, 16.3)
    draw_arrow(ax, 6, 15.5, 6, 14.8)
    draw_arrow(ax, 6, 14, 6, 13.3)
    draw_arrow(ax, 6, 12.5, 6, 11.8)
    draw_arrow(ax, 6, 11, 6, 10.3)
    draw_arrow(ax, 6, 9.5, 6, 8.5)
    
    draw_arrow(ax, 7, 7.5, 8, 6.8, "Yes")
    draw_arrow(ax, 5, 7.5, 3, 6.8, "No")
    
    draw_arrow(ax, 9.5, 6, 9.5, 5.3)
    draw_arrow(ax, 9.5, 4.5, 9.5, 3.8)
    
    draw_arrow(ax, 2, 4.5, 2, 3) # Just to align
    draw_arrow(ax, 2, 4, 5, 1.5)
    draw_arrow(ax, 9.5, 3, 7, 1.5)
    
    plt.tight_layout()
    plt.savefig("uml_activity_diagram.jpg", dpi=300, bbox_inches='tight')
    plt.close()

def create_class():
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    plt.title("UML Class Diagram: Software Structure", fontsize=20, fontweight='bold', pad=20)
    
    # Classes
    # ClassifierService
    draw_box(ax, 1, 9, 3, 2, "ClassifierService\n--\n- model: ResNet50\n- categories: List\n--\n+ predict(image): dict", color='#E1F5FE')
    
    # EmbeddingService
    draw_box(ax, 1, 5, 3, 2, "DINOv2EmbeddingService\n--\n- model: DINOv2\n- dimension: int\n--\n+ embed_image(image): np.array", color='#E1F5FE')
    
    # FAISSIndex
    draw_box(ax, 5, 7, 3, 2, "FAISSGalleryIndex\n--\n- index: IndexFlatIP\n- metadata: List\n--\n+ search(vec, k): list\n+ save()\n+ load()", color='#E8F5E9')
    
    # VisualSearchService
    draw_box(ax, 5, 3, 3, 2, "VisualSearchService\n--\n- faiss_index: FAISSGalleryIndex\n--\n+ search_design(): list", color='#E8F5E9')
    
    # RAGRetriever
    draw_box(ax, 9, 7, 3, 2, "RAGMetadataRetriever\n--\n- collection: Collection\n- client: PersistentClient\n--\n+ get_by_design_id(): dict", color='#F3E5F5')
    
    # RAGGenerator
    draw_box(ax, 9, 3, 3, 2, "RAGGeneratorService\n--\n- api_key: str\n- model: str\n--\n+ generate_summary(): str\n+ generate_answer(): str", color='#F3E5F5')
    
    # RejectionService
    draw_box(ax, 5, 1, 3, 1.5, "RejectionDecisionService\n--\n--\n+ evaluate(): dict", color='#FFF3E0')
    
    # Arrows (Dependencies)
    draw_arrow(ax, 4, 9.5, 5, 8.5) # Classifier -> FAISS
    draw_arrow(ax, 4, 6, 5, 7.5)    # Embedding -> FAISS
    draw_arrow(ax, 6.5, 7, 6.5, 5)  # FAISSIndex -> VisualSearch
    draw_arrow(ax, 8, 4, 9, 4)      # Search -> RAGGen
    draw_arrow(ax, 8, 3.5, 9, 7.5)  # Search -> RAGRet
    draw_arrow(ax, 6.5, 3, 6.5, 2)  # Search -> Rejection
    
    plt.tight_layout()
    plt.savefig("uml_class_diagram.jpg", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_architecture()
    create_activity()
    create_class()
    print("Diagrams generated successfully.")
