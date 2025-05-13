from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
import json

app = Flask(__name__, static_folder='.')
CORS(app)

# Khởi tạo Chroma với persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Instantiate image loader helper
image_loader = ImageLoader()

# Instantiate multimodal embedding function
multimodal_ef = OpenCLIPEmbeddingFunction()

# Lấy collection hoặc tạo mới
image_collection = chroma_client.get_or_create_collection(
    name="image_search", 
    embedding_function=multimodal_ef, 
    data_loader=image_loader
)


@app.route('/')
def serve_frontend():
    return render_template('frontend.html')


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/search', methods=['POST'])
def search_image():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Debug: In ra truy vấn
    print(f"Đang tìm kiếm: '{query}'")
    
    # Tìm 10 kết quả gần nhất (hoặc ít hơn nếu không đủ)
    collection_count = image_collection.count()
    n_results = min(100, collection_count)
    
    if collection_count == 0:
        return jsonify({'results': [], 'message': 'Không có ảnh trong collection'}), 200
    
    results = image_collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "distances", "uris"]
    )
    
    # Debug: In ra kết quả raw từ chroma
    print(f"Kết quả từ ChromaDB: {json.dumps(results, default=str)[:200]}...")

    matches = []
    if results and 'ids' in results and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            match = {
                'id': results['ids'][0][i],
                'filename': results['metadatas'][0][i]['filename'],
                'score': float(results['distances'][0][i]) if 'distances' in results else 0.0,
                'uri': results['uris'][0][i] if 'uris' in results else None
            }
            matches.append(match)

    # Debug: In ra số lượng kết quả
    print(f"Đã tìm thấy {len(matches)} kết quả")
    
    return jsonify({'results': matches})


@app.route('/stats', methods=['GET'])
def get_stats():
    """Trả về thống kê về collection"""
    try:
        count = image_collection.count()
        return jsonify({
            'count': count,
            'status': 'ok'
        })
    except Exception as e:
        return jsonify({
            'count': 0,
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    # Hiển thị thông tin về collection trước khi chạy
    try:
        count = image_collection.count()
        print(f"✅ ChromaDB đã có {count} ảnh trong collection 'image_search'")
        
        # Nếu collection trống, thông báo chạy load_images.py trước
        if count == 0:
            print("⚠️ Collection trống! Vui lòng chạy load_images.py trước")
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra collection: {e}")
    
    app.run(debug=True)