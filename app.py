from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
import secrets
import re
import sqlite3


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# -------------------------- 上传配置 --------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# -------------------------- 数据库初始化（自动创建） --------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()


# -------------------------- 注册/登录规则校验 --------------------------
def check_username(username):
    if len(username) < 4 or len(username) > 20:
        return False
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False
    return True


def check_password(password):
    if len(password) < 6 or len(password) > 20:
        return False
    return True


def username_exists(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    res = c.fetchone()
    conn.close()
    return res is not None


def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()


def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    res = c.fetchone()
    conn.close()
    if res and res[0] == password:
        return True
    return False


# -------------------------- 登录拦截（所有页面必须登录） --------------------------
@app.before_request
def check_login():
    if request.path in ['/login', '/register', '/static']:
        return
    if not session.get('logged_in'):
        return redirect('/login')


# -------------------------- 登录/注册/退出 --------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    username = request.form.get('username')
    password = request.form.get('password')

    if verify_user(username, password):
        session['logged_in'] = True
        session['user'] = username
        return redirect('/')
    else:
        return render_template('login.html', msg="账号或密码错误")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    username = request.form.get('username')
    password = request.form.get('password')

    if not check_username(username):
        return render_template('register.html', msg="账号必须4-20位，仅支持字母、数字、下划线")
    if not check_password(password):
        return render_template('register.html', msg="密码必须6-20位")
    if username_exists(username):
        return render_template('register.html', msg="账号已被注册")

    add_user(username, password)
    return redirect('/login')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ================== LDA 主题聚类分析（高级功能） ==================
@app.route('/lda')
def lda_page():
    return render_template('lda.html')

@app.route('/api/lda')
def api_lda():
    fn = request.args.get('f')
    if not fn:
        return jsonify({'error': '未上传文件'})

    df = get_df(fn)
    texts = []

    for _, row in df.iterrows():
        title = str(row.get('Title-题名', ''))
        kw = str(row.get('Keyword-关键词', ''))
        text = (title + ' ' + kw).strip()
        if text:
            texts.append(text)

    # LDA 建模
    vectorizer = CountVectorizer(max_features=200, stop_words=['的', '和', '在', '等', '与', '对'])
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[-15:]]
        topics.append({
            'topic': f'主题 {idx+1}',
            'words': top_features
        })

    return jsonify({'topics': topics})



# -------------------------- 你原来的所有路由（保持不变） --------------------------
@app.route('/')
def index():
    return render_template('citespace.html')


@app.route('/metrics')
def metrics_page():
    return render_template('metrics.html')


@app.route('/network')
def network_page():
    return render_template('network.html')


@app.route('/timeline')
def timeline_page():
    return render_template('timeline.html')


@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html')


@app.route('/list')
def list_page():
    return render_template('list.html')



# -------------------------- 工具函数（必须保留！） --------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 修复：补全 get_df 函数，解决500错误
def get_df(filename):
    import chardet
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # 自动检测编码，兼容知网CSV的GBK/UTF-8
    with open(path, 'rb') as f:
        enc = chardet.detect(f.read())['encoding'] or 'gbk'
    try:
        return pd.read_csv(path, encoding=enc, on_bad_lines='skip')
    except:
        return pd.read_csv(path, encoding='gbk', on_bad_lines='skip')


# -------------------------- 你原来的上传接口 --------------------------
# -------------------------- 上传接口（修复版） --------------------------
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未检测到文件，请重新选择'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '仅支持CSV格式文件'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 验证文件是否能正常读取、列名是否正确
        df = get_df(filename)
        required_cols = ['Title-题名', 'Author-作者', 'Keyword-关键词', 'PubTime-发表时间']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return jsonify({
                'error': f'CSV缺少必需列：{", ".join(missing_cols)}，请使用知网导出的标准格式'
            }), 400

        return jsonify({'success': True, 'filename': filename, 'msg': '上传成功！可使用所有功能'})

    except Exception as e:
        # 捕获所有异常，返回具体错误（方便排查）
        return jsonify({'error': f'解析失败：{str(e)[:100]}'}), 500


# -------------------------- 网络数据接口（修复版，带连线！） --------------------------
from collections import defaultdict, Counter
import re

#1.
@app.route('/api/network')
def api_network():
    fn = request.args.get('f')
    if not fn:
        return jsonify({'error': '未上传文件'}), 400

    df = get_df(fn)

    # ========== 1. 作者合作网络（带连线） ==========
    author_nodes = []
    author_links = []
    author_counter = Counter()
    author_cooccur = defaultdict(Counter)  # 作者共现计数

    # ========== 2. 关键词共现网络（带连线） ==========
    kw_nodes = []
    kw_links = []
    kw_counter = Counter()
    kw_cooccur = defaultdict(Counter)  # 关键词共现计数

    # 遍历每篇文献，统计共现关系
    for _, row in df.iterrows():
        # -------------------------- 处理作者 --------------------------
        author_str = str(row.get('Author-作者', ''))
        author_list = re.split(r'[;；\s]', author_str)
        author_list = [a.strip() for a in author_list if a.strip() and len(a.strip()) > 1]

        # 统计作者出现次数（节点大小）
        for a in author_list:
            author_counter[a] += 1

        # 统计作者共现（连线权重）
        for i in range(len(author_list)):
            for j in range(i + 1, len(author_list)):
                a1, a2 = author_list[i], author_list[j]
                if a1 > a2:
                    a1, a2 = a2, a1
                author_cooccur[a1][a2] += 1

        # -------------------------- 处理关键词 --------------------------
        kw_str = str(row.get('Keyword-关键词', ''))
        kw_list = re.split(r'[;；\s]', kw_str)
        kw_list = [k.strip() for k in kw_list if k.strip() and k.lower() != 'nan']

        # 统计关键词出现次数（节点大小）
        for k in kw_list:
            kw_counter[k] += 1

        # 统计关键词共现（连线权重）
        for i in range(len(kw_list)):
            for j in range(i + 1, len(kw_list)):
                k1, k2 = kw_list[i], kw_list[j]
                if k1 > k2:
                    k1, k2 = k2, k1
                kw_cooccur[k1][k2] += 1

    # -------------------------- 生成作者节点 & 连线 --------------------------
    # 取TOP30高频作者（避免节点过多）
    top_authors = [a for a, c in author_counter.most_common(30)]
    author_name_to_id = {name: idx for idx, name in enumerate(top_authors)}

    # 生成节点
    author_nodes = [{'name': name, 'symbolSize': c * 3 + 10, 'value': c} for name, c in author_counter.most_common(30)]

    # 生成连线（只保留TOP30作者之间的共现）
    for a1, counter in author_cooccur.items():
        if a1 not in author_name_to_id:
            continue
        for a2, weight in counter.items():
            if a2 not in author_name_to_id:
                continue
            author_links.append({
                'source': author_name_to_id[a1],
                'target': author_name_to_id[a2],
                'value': weight,
                'lineStyle': {'width': weight}
            })

    # -------------------------- 生成关键词节点 & 连线 --------------------------
    # 取TOP50高频关键词（过滤nan）
    top_kws = [k for k, c in kw_counter.most_common(50) if k.lower() != 'nan']
    kw_name_to_id = {name: idx for idx, name in enumerate(top_kws)}

    # 生成节点
    kw_nodes = [{'name': name, 'symbolSize': c * 2 + 8, 'value': c} for name, c in kw_counter.most_common(50) if
                name.lower() != 'nan']

    # 生成连线（只保留TOP50关键词之间的共现）
    for k1, counter in kw_cooccur.items():
        if k1 not in kw_name_to_id or k1.lower() == 'nan':
            continue
        for k2, weight in counter.items():
            if k2 not in kw_name_to_id or k2.lower() == 'nan':
                continue
            kw_links.append({
                'source': kw_name_to_id[k1],
                'target': kw_name_to_id[k2],
                'value': weight,
                'lineStyle': {'width': weight}
            })

    # 返回完整数据（节点+连线）
    return jsonify({
        'author_nodes': author_nodes,
        'author_links': author_links,
        'kw_nodes': kw_nodes,
        'kw_links': kw_links,
        'topk': [{'name': k, 'value': c} for k, c in kw_counter.most_common(50) if k.lower() != 'nan']
    })


# 2. 统计数据接口（高产作者/文献来源/年度趋势）
from collections import Counter
import re


@app.route('/api/stats')
def api_stats():
    fn = request.args.get('f')
    if not fn:
        return jsonify({
            "year": {},
            "authors": {},
            "sources": {},
            "citations": {}
        })

    df = get_df(fn)

    from collections import Counter
    import re

    year_counter = Counter()
    author_counter = Counter()
    source_counter = Counter()
    citation_counter = Counter()

    for _, row in df.iterrows():
        # 年份
        pub_time = str(row.get('PubTime-发表时间', ''))
        year_match = re.search(r'(\d{4})', pub_time)
        if year_match:
            year = year_match.group(1)
            year_counter[year] += 1

        # 作者
        author_str = str(row.get('Author-作者', '')).strip()
        if author_str and author_str.lower() != 'nan':
            authors = re.split(r'[;；]', author_str)
            for a in authors:
                a = a.strip()
                if a:
                    author_counter[a] += 1

        # 来源
        source = str(row.get('Source-文献来源', '')).strip()
        if source and source.lower() != 'nan':
            source_counter[source] += 1

        # 被引文献（按标题统计被引热度）
        title = str(row.get('Title-题名', '')).strip()
        if title and title.lower() != 'nan':
            citation_counter[title] += 1

    return jsonify({
        "year": dict(year_counter),
        "authors": dict(author_counter),
        "sources": dict(source_counter),
        "citations": dict(citation_counter)
    })


# 3. 文献列表接口（检索用）
@app.route('/api/papers')
def api_papers():
    fn = request.args.get('f')
    wd = request.args.get('wd', '')
    if not fn:
        return jsonify([])

    df = get_df(fn)
    df = df.where(pd.notnull(df), '')

    if wd:
        df = df[
            df['Title-题名'].str.contains(wd, na=False) |
            df['Author-作者'].str.contains(wd, na=False) |
            df['Keyword-关键词'].str.contains(wd, na=False)
            ]

    return jsonify(df.to_dict(orient='records'))




if __name__ == '__main__':
    app.run(debug=True)