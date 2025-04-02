#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import esm
import torch
import os
import sys



# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 创建Flask应用，并指定模板目录
app = Flask(__name__)



def resource_path(relative_path):
    """获取资源的绝对路径"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def predict_peptide(sequence_list):
    try:
        # 预处理数据
        peptide_sequence_list = []
        for seq in sequence_list:
            format_seq = [seq, seq]
            tuple_sequence = tuple(format_seq)
            peptide_sequence_list.append(tuple_sequence)
            
        # 加载ESM-2模型
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        # 加载数据
        data = peptide_sequence_list
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        # 提取表示
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        token_representations = results["representations"][6]
        
        # 生成序列表示
        sequence_representations = []
        for i, token_len in enumerate(batch_lens):
            each_seq_rep = token_representations[i, 1:token_len - 1].mean(0).tolist()  
            sequence_representations.append(each_seq_rep)

        embedding_results = pd.DataFrame(sequence_representations)
        
        # 加载SVM模型
        model_path = resource_path('svm.pkl')
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        # 预测
        predictions = loaded_model.predict(embedding_results)
        decision_values = loaded_model.decision_function(embedding_results)
        probabilities = 1 / (1 + np.exp(-decision_values))
        
        # 处理结果
        result = []
        for i in range(len(predictions)):
            if predictions[i] == 0:
                result.append({'sequence': sequence_list[i], 'prediction': 'non_active', 'probability': 'N'})
            elif predictions[i] == 1:
                result.append({'sequence': sequence_list[i], 'prediction': 'active', 'probability': f"{probabilities[i]:.6f}"})
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sequences = data.get('sequences', [])
    
    if not sequences:
        return jsonify({'error': 'Please provide sequences'})
    
    results = predict_peptide(sequences)
    return jsonify(results)

# 添加CORS支持
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    # 获取Heroku分配的端口
    port = int(os.environ.get('PORT', 5000))
    # 设置调试模式
    app.debug = False  # 生产环境关闭调试模式
    # 设置主机和端口
    app.run(host='0.0.0.0', port=port)

