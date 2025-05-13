import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

# Kiểm tra thư mục images tồn tại
image_folder = "./images"
if not os.path.exists(image_folder):
    print(f"⚠️ Thư mục {image_folder} không tồn tại!")
    os.makedirs(image_folder)
    print(f"✅ Đã tạo thư mục {image_folder}")

# Kiểm tra và in ra các file ảnh thực sự tồn tại
existing_files = os.listdir(image_folder) if os.path.exists(image_folder) else []
print(f"Các file hiện có trong thư mục {image_folder}: {existing_files}")

# Khởi tạo Chroma với persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Instantiate image loader helper
image_loader = ImageLoader()

# Instantiate multimodal embedding function
multimodal_ef = OpenCLIPEmbeddingFunction()

# Xóa collection cũ nếu có
try:
    chroma_client.delete_collection("image_search")
    print("Đã xóa collection cũ")
except:
    print("Không có collection cũ để xóa")

# Tạo collection mới với embedding function và data loader
image_collection = chroma_client.create_collection(
    name="image_search", 
    embedding_function=multimodal_ef, 
    data_loader=image_loader
)

# Danh sách các file ảnh cần thêm vào
image_files = [f for f in existing_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if not image_files:
    print(f"⚠️ Không tìm thấy file ảnh nào trong thư mục {image_folder}!")
else:
    # Tạo đường dẫn đầy đủ cho mỗi file ảnh
    image_paths = [os.path.join(image_folder, file) for file in image_files]
    
    # Tạo IDs cho các ảnh
    image_ids = [str(i) for i in range(len(image_files))]
    
    # Tạo metadata cơ bản cho mỗi ảnh
    image_metadatas = [{"filename": file} for file in image_files]
    
    # Thêm ảnh vào collection
    print("Đang thêm ảnh vào ChromaDB...")
    image_collection.add(
        ids=image_ids,
        uris=image_paths,
        metadatas=image_metadatas
    )
    
    print(f"✅ Đã thêm {len(image_files)} ảnh vào ChromaDB!")

# Kiểm tra collection
collection_count = image_collection.count()
print(f"Số lượng ảnh trong collection: {collection_count}")

# Thử truy vấn với một văn bản đơn giản để kiểm tra
if collection_count > 0:
    print("\nKiểm tra tìm kiếm với từ khóa 'cat':")
    results = image_collection.query(
        query_texts=["cat"],
        n_results=min(3, collection_count),
        include=["metadatas", "distances", "uris"]
    )
    
    if results and len(results["ids"][0]) > 0:
        print(f"Tìm thấy {len(results['ids'][0])} kết quả:")
        for i in range(len(results["ids"][0])):
            print(f"  {results['metadatas'][0][i]['filename']} (Score: {results['distances'][0][i]})")
    else:
        print("Không tìm thấy kết quả!")