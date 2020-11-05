# Program: SDPs-for-PTOs-and-genes
# Author：生信1702 邵燕涵 2017317220205
# Software：PyCharm 2020.1.1 (Community Edition)
# Enviroment：python3.7

pip install /Users/macbookair/Downloads/en_core_web_sm-2.2.5.tar.gz
import spacy
import networkx as nx
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import os

#nltk.__path__=['/Users/macbookair/opt/anaconda3/lib/python3.7/site-packages/nltk']

def read_pto(pto_file):
    """
    读取PTO文件 存为字典
    key: pto term id
    value: name and synonym
    :param pto_file: PTO 文件
    :return: 字典
    """
    opt_dic = defaultdict(list)  #存储PTOid，name，sym（近义词）。一个位置存放三个地方
    count_term = 0  #PTO数量
    count_word = 0  #包括名字和其近义词的数量
    with open(file=pto_file,mode='r',encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if l.startswith('id'): #id开头的话，存储
                count_term += 1
                id = l.split()[1]  #去除PTOid
            if l.startswith('name'):
                # 读取名字
                count_word += 1
                name = ' '.join(l.split(' ')[1:])
                if '(' in name:
                    name = name[:name.find('(') - 1 ]  #取出name，去除（）
                opt_dic[id].append(name)
            if l.startswith('synonym'):
                # 正则匹配读取同义名
                count_word += 1
                pattern = re.compile('"(.*)"')
                synonym = pattern.findall(l)[ 0 ]   #讲列表转为字符串
                pattern_ba = re.compile(r'[(].*?[)]')  #取出括号内容
                backets = pattern_ba.findall(l)
                if synonym.startswith('('):  #讲除第一个之后的括号都删除
                    for ba in backets[ 1: ]:
                        synonym = synonym.replace(ba, '')
                        synonym = synonym.strip(' ')
                else:  #如果开头没有括号，则所有括号都删除
                    for ba in backets:
                        synonym = synonym.replace(ba, '')
                        synonym = synonym.strip(' ')
                opt_dic[id].append(synonym)
    print('一共 {0} terms, 包含 {1} 个名字及同义名.'.format(count_term, count_word))
    return opt_dic

def abs_read(abstract_file: str):
    # 读取摘要 因为文件不大就全部存进内存了
    title_abstract_list = []
    count_abs = 0  #统计读到第几篇摘要
    with open(file=abstract_file,mode='r',encoding="utf-8") as f:
        line = f.readline()
        for line in f:
            l = line.strip().split('\t')  #按照制表符分隔
            try:
                pmid = l[0]
                title = l[1]  #Title
                abs = l[5]   #Abstract
                title_abstract_list.append((pmid,title, abs))
                count_abs += 1
            except:   #处理异常
                print(l)
                continue
    print('共 {0} 篇摘要.'.format(count_abs))
    return title_abstract_list

# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#词性分析后进行词性还原
#输入文件：word_list 单词列表
#输出文件：list_words_lemmatizer单词词形还原列表,list_sents_lemmatizer句子词形还原
def tag_lemmatizer(word_list):
    tagged_words = pos_tag(word_list)
    wnl = WordNetLemmatizer()
    list_words_lemmatizer = []
    for tag in tagged_words:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        list_words_lemmatizer.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    sents_lemmatizer=(" ".join(list_words_lemmatizer))   #分句词性还原
    return list_words_lemmatizer,sents_lemmatizer

#对PTO进行词形还原
def PTO_lem(pto_dic):
    pto_lem_dic = defaultdict(list)
    for term,name_list in pto_dic.items():
        for name in name_list:
            name_lem,names_lem=tag_lemmatizer(word_tokenize(name))  #PTO词形还原
            pto_lem_dic[term].append(names_lem)
    return pto_lem_dic

def save_result(ta_list: list, pto_dic: dict, pto_lem_dic: dict, out_file: str):
    # 硬匹配搜索
    print('PTO search running.')
    wf = open(out_file, 'w' , encoding="utf-8")
    wf.write('PTO\tsentence\tpmid\tterm\n')   #输出文件标题行
    count = 0#记录载入的摘要
    count_result = 0
    for ta in ta_list:  #摘要文件
        count += 1
        if count % 600 == 0:
            print('{0}/{1} abstracts process done.'.format(count, len(ta_list)))
        pmid = ta[0]
        abs = ta[2]  #abstract
        sent_list = [i for i in sent_tokenize(abs)]  #对摘要进行句子的分句
        for sent in sent_list:
            # 迭代每个句子
            word_list = word_tokenize(sent)
            # 分词
            words_lem,sent_lem=tag_lemmatizer(word_list)
            #摘要词性还原
            for term, name_list in pto_lem_dic.items():
                #{'num1': 'Tom', 'num2': 'Lucy'}
                #dict_items([('num1', 'Tom'), ('num2', 'Lucy')])
                for name in name_list:
                    # 迭代PTO名字和同义名
                    if len(name.split(' ')) > 1:  #用所有name和sym匹配
                        # 如果PTO名字是一个词组 转换为小写直接在句子里匹配
                        names=word_tokenize(name)
                        a=0
                        word = [i.lower() for i in words_lem]  #摘要词转小写
                        for i in names:
                            a+=1
                            if i not in stop_words and i.lower() in word:
                            #去除停用词
                            #if i.lower() in words_lem:  #不去停用词
                                if a==len(names):
                                    wf.write('{0}\t{1}\t{2}\t{3}\n'. \
                                             format(term, sent, pmid, name))
                                    count_result += 1
                            else:
                                break
                    else:   #这里未变化大小写
                        if name in words_lem:  #不转小写，避免去除An这类关键词
                            # 如果PTO名字是单个单词 在分词列表中匹配
                            # 这样做是基于经验的做法 会更加准确
                                wf.write('{0}\t{1}\t{2}\t{3}\n'. \
                                     format(term, sent, pmid, name))
                                count_result += 1
    print('共找到 {0} 个句子包含pto条目.'.format(count_result))
    wf.close()

pto_file = '/Users/macbookair/Downloads/TO_basic.obo'
abs_file = '/Users/macbookair/Downloads/reference_PMID.match.table.txt'
#out_file = '/store/yhshao/result_pto.txt'
with open('/Users/macbookair/Downloads/stop_words.txt','r',encoding='utf-8') as fr:
   stop_words=fr.read().split('\n') #将停用词读取到列表里

pto_dic = read_pto(pto_file)   #读取PTO文件
# print(pto_dic)
ta_list = abs_read(abs_file)  #读取摘要文件
#pto_lem_dic=PTO_lem(pto_dic)   #获取PTO词形还原
#save_result(ta_list, pto_dic, pto_lem_dic,out_file)  #寻找PTO

nlp=spacy.load("en_core_web_sm")

f=open('/Users/macbookair/Downloads/match.txt','r')
x=[]#存储1600个共显性结果
for i in f:
    l=i.strip()
    ll=l.split('\t')
    x.append(ll)
f.close()
x.remove(x[0])
x=sorted(x,key=lambda x:x[0])

abs_list=abs_read('/Users/macbookair/Downloads/reference_PMID.match.table.txt')
abs_list=sorted(abs_list,key=lambda x:x[0])

b=list()
for i in x:
    b.append(i[0])
temp = []
for item in b:
    if not item in temp:
        temp.append(item)
"""
c={}
for i in temp:
    for j in abs_list:
        if i==j[0]:
            sent_list = [m for m in sent_tokenize(j[2])]
            c[i]=sent_list
"""
import json

file = open('/Users/macbookair/Downloads/divided.txt', 'r')
js = file.read()
dic = json.loads(js)
file.close()


f=open('/Users/macbookair/Downloads/result_pto.txt','r')
y=[]#存储匹配到PTO的句子等信息
for i in f:
    l=i.strip()
    ll=l.split('\t')
    y.append(ll)
f.close()
y.remove(y[0])
y=sorted(y,key=lambda x:x[2])

wf = open('/Users/macbookair/Downloads/onesent.txt', 'w' , encoding="utf-8")
wf.write('PMID\tgenename\tPTOterm\tsent\n')
count=0
for i in y:
    for j in x:
        if i[2]==j[0]:
            if ',' in j[2]:
                gname=j[2].split('[\'')[1].split('\']')[0].split('\'')
                for m in gname[::2]:
                    if m in i[1]:
                        wf.write('{0}\t{1}\t{2}\t{3}\n'.format(j[0], m, i[3], i[1]))
                        count=count+1
            else:
                gname=j[2].split('[\'')[1].split('\']')[0]
                if gname in i[1]:
                    wf.write('{0}\t{1}\t{2}\t{3}\n'.format(j[0], gname, i[3], i[1]))
                    count=count+1
print('一共匹配到{0}条结果(gene与PTOterm同句)'.format(count))#一共匹配到2045条结果(gene与PTOterm同句)
wf.close()

f=open('/Users/macbookair/Downloads/onesent.txt','r')
z=[]#存储匹配到PTOterm与gene同句的信息
for i in f:
    l=i.strip()
    ll=l.split('\t')
    z.append(ll)
f.close()
z.remove(z[0])

z1=[]
count=0
for i in z:
    if not i in z1:
        z1.append(i)
        count=count+1
print('去重后一共{0}条结果(gene与PTOterm同句)'.format(count))#去重后一共594条结果(gene与PTOterm同句)

nlp=spacy.load("en_core_web_sm")
wf = open('/Users/macbookair/Downloads/SDP.txt', 'w' , encoding="utf-8")
wf.write('genename\trelation\tPTOterm\tdistance\n')
getexcept=[]
fail_count=0
success_count=0
relation=()
for i in z1:
    doc=nlp(i[3])
    edges=[]
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
    graph=nx.Graph(edges)
    entity1=i[1].lower()
    entity2=i[2].lower()
    try:
        if ' ' in i[2]:
            s=0
            for j in i[2].split(' '):
                s=s+nx.shortest_path_length(graph,source=entity1,target=j)
            distance=int((s/len(i[2].split(' ')))+0.5)
            for token in doc:
                if token.dep_ == 'ROOT':
                    relation=(token.head.text)
            wf.write('{0}\t{1}\t{2}\t{3}\n'.format(i[1], relation, i[2], distance))
            success_count+=1
        else:
            distance=nx.shortest_path_length(graph,source=entity1,target=entity2)
            for token in doc:
                if token.dep_ == 'ROOT':
                    relation=(token.head.text)
            wf.write('{0}\t{1}\t{2}\t{3}\n'.format(i[1], relation, i[2], distance))
            success_count+=1
    except:
        getexcept.append(i)
        fail_count+=1
        continue
wf.close()

f=open('/Users/macbookair/Downloads/SDP.txt','r')
w=[]#存储SDP的信息
for i in f:
    l=i.strip()
    ll=l.split('\t')
    w.append(ll)
f.close()
w.remove(w[0])

w1=[]
count=0
for i in w:
    if not i in z1:
        w1.append(i)
        count=count+1
print('去重后一共{0}条结果(SDP)'.format(count))#去重后一共594条结果(SDP)

for i in pto_dic.keys():
    if 'SPS' in pto_dic[i]:
        print(pto_dic[i])


wf = open('/Users/macbookair/Downloads/SDP(excepet).txt', 'w' , encoding="utf-8")
wf.write('genename\trelation\tPTOterm\tdistance\n')
fail_count=0
success_count=0
relation=()
getexcept2=[]
for result in getexcept:

    words = word_tokenize(result[3])
    delete = 0
    for item in words:
        if item in (',', '.'):
            words.remove(item)
        elif item == '(':
            for i in words[words.index(item):]:
                if i == ')':
                    words.remove(i)
                    break
                words.remove(i)

    if result[1] not in words:
        paste = []
        count = -1
        for i in words:
            pasted=[]
            paste.append(i)
            count+=1
            pasted=' '.join(paste)
            if result[1] in pasted:
                words[count]=result[1]
                paste=[]
        wordafter = []
        for i in words:
            if i not in (',', '.'):
                wordafter.append(i)
        sent = ' '.join(wordafter)

    if '-' in result[1]:
        count=-1
        for i in words:
            count+=1
            if i==result[1]:
                words[count]=''.join(i.split('-'))
        result[1]=''.join(result[1].split('-'))
        wordafter = []
        for i in words:
            if i not in (',', '.'):
                wordafter.append(i)
        sent = ' '.join(wordafter)

    if '/' in result[1]:
        count=-1
        for i in words:
            count+=1
            if i==result[1]:
                words[count]=''.join(i.split('/'))
        result[1]=''.join(result[1].split('/'))
        wordafter = []
        for i in words:
            if i not in (',', '.'):
                wordafter.append(i)
        sent = ' '.join(wordafter)

    if ';' in result[1]:
        count=-1
        for i in words:
            count+=1
            if i==result[1]:
                words[count]=''.join(i.split(';'))
        result[1]=''.join(result[1].split(';'))
        wordafter = []
        for i in words:
            if i not in (',', '.'):
                wordafter.append(i)
        sent = ' '.join(wordafter)

    mis=[]
    for i in range(len(result[2].split(' '))):
        if result[2].split(' ')[i] not in words:
            mis.append(i)
    if mis!=[]:
        if len(mis)>1:
            for i in range(len(mis)):
                paste=[]
                count=-1
                for j in words:
                    pasted=[]
                    paste.append(j)
                    count += 1
                    pasted=' '.join(paste)
                    if result[2].split(' ')[mis[i]] in pasted:
                        words[count] = result[2].split(' ')[mis[i]]
                        paste=[]
            wordafter = []
            for i in words:
                if i not in (',', '.'):
                    wordafter.append(i)
            sent = ' '.join(wordafter)


        paste=[]
        count=-1
        for i in words:
            pasted=[]
            paste.append(i)
            count+=1
            pasted=' '.join(paste)
            if result[2].split(' ')[mis[0]] in pasted:
                words[count]=result[2].split(' ')[mis[0]]
                paste=[]
        wordafter = []
        for i in words:
            if i not in (',', '.'):
                wordafter.append(i)
        sent = ' '.join(wordafter)

    if '-' in result[2]:
        count=-1
        for i in words:
            count+=1
            if i==result[2]:
                words[count]=''.join(i.split('-'))
        result[2]=''.join(result[2].split('-'))
        wordafter = []
        for i in words:
            if i not in (',', '.'):
                wordafter.append(i)
        sent = ' '.join(wordafter)

    doc = nlp(sent)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_), '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    entity1 = result[1].lower()
    entity2 = result[2].lower()
    try:
        s = 0
        for j in entity2.split(' '):
            s = s + nx.shortest_path_length(graph, source=entity1, target=j)
        distance = int((s / len(entity2.split(' '))) + 0.5)
        for token in doc:
            if token.dep_ == 'ROOT':
                relation = (token.head.text)
        wf.write('{0}\t{1}\t{2}\t{3}\n'.format(entity1, relation, entity2, distance))
        success_count += 1
    except:
        getexcept2.append(result)
        fail_count += 1
        continue
wf.close()
