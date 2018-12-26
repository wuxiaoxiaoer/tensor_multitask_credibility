import os
import nltk
from utils.readFileUtil import readFile
from gensim.models import word2vec

def getliar_text_metadata_vectors(dir, start, length, is_maintask):
    file = []
    # dir = 'data/liar_train.tsv'
    # fake新闻中所有信息
    file = readFile(dir, start, length)
    # 文本2中去除停用词的分词
    article_list = []
    # 1.获得1标注列表
    labels = []
    # 获得听用词列表
    en_stop_words = open('tools/en_stop_words.txt', encoding='utf-8')
    # 获得word2vec Model
    model = word2vec.Word2Vec.load("tools/text8.model")

    # # liar3主题词, 共143个
    # topics_file = open('tools/statis_3topics.txt', encoding='utf-8')
    # # liar6州, 共86个
    # states_file = open('tools/statis_6state.txt', encoding='utf-8')
    # # liar7党派, 共23个
    # community_file = open('tools/statis_7community.txt', encoding='utf-8')
    # topics_list = []
    # for line in topics_file:
    #     line = line.replace('\n', '')
    #     topics_list.append(line)
    # states_list = []
    # for line in states_file:
    #     line = line.replace('\n', '')
    #     states_list.append(line)
    # community_list = []
    # for line in community_file:
    #     line = line.replace('\n', '')
    #     community_list.append(line)
    # # 获得8信用历史数据
    # credData = []
    # # 3主题_7党派_6州数据
    # topic_comm_state = []

    for line in file:
        i = 0
        # topic_comm_state_line = []
        # cred = [0.5] * 200
        # topics = [0.5] * 180
        # community = [0.5] * 30
        # states = [0.5] * 90
        for col in line:
            col = col.strip().lower()
            if col == '':
                i += 1
                continue
            # col[2] i=2: statement文本词向量特征
            if i == 2:
                tokens = nltk.word_tokenize(col)
                clear_sw_tokens = []
                for word in tokens:
                    if word not in en_stop_words:
                        clear_sw_tokens.append(word)
                article_list.append(clear_sw_tokens)
            # # 3主题 143
            # if i == 3:
            #     col_split = col.split(',')
            #     print("主题 1==3：")
            #     print(col_split)
            #     for item in col_split:
            #         if item in topics_list:
            #             index = topics_list.index(item)
            #             topics[index] = 1
            #     pass
            # # 6州 86
            # if i == 6:
            #     if item in states_list:
            #         index = states_list.index(col)
            #         states[index] = 1
            #     pass
            # # 7党派 23
            # if i == 7:
            #     if item in community_list:
            #         index = community_list.index(col)
            #         community[index] = 1
            #     pass
            # # 8用户信用
            # if i == 8:
            #     cred[0] = int(col)
            # if i == 9:
            #     cred[1] = int(col)
            # if i == 10:
            #     cred[2] = int(col)
            # if i == 11:
            #     cred[3] = int(col)
            # if i == 12:
            #     cred[4] = int(col)
            # 1标签
            if i == 1:
                if is_maintask == True:
                    label = []
                    if col == 'barely-true':
                        label = [1, 0, 0, 0, 0, 0]
                    if col == 'false':
                        label = [0, 1, 0, 0, 0, 0]
                    if col == 'half-true':
                        label = [0, 0, 1, 0, 0, 0]
                    if col == 'mostly-true':
                        label = [0, 0, 0, 1, 0, 0]
                    if col == 'pants-fire':
                        label = [0, 0, 0, 0, 1, 0]
                    if col == 'true':
                        label = [0, 0, 0, 0, 0, 1]
                    labels.append(label)
                # 如果是多任务
                if is_maintask == False:
                    if col == 'true':
                        label = [1, 0]
                    else:
                        label = [0, 1]
                    labels.append(label)
            i += 1
        # credData.append(cred)
        # topic_comm_state_line.extend(topics)
        # topic_comm_state_line.extend(community)
        # topic_comm_state_line.extend(states)
        # topic_comm_state.append(topic_comm_state_line)

    # 2.获得文章的词向量特征
    article_vec = []
    for single in article_list:
        single_vec = []
        word_i = 0
        # 1)限制50个词，如果太多舍弃; 2) 不在embeddings中补充0; 3)如果不够50，补充0。
        for word in single:
            word_i += 1
            # 1) 限制50个词，如果太多舍弃;
            if word_i <= 50:
                word_vec = []
                try:
                    word_vec = model[word]
                # 2) 不在embeddings中补充0;
                except KeyError:
                    word_vec = [0.0]*200
                    # print('word ' + word + ' not in vocabulary')
                    pass
                single_vec.append(word_vec)
                # print(word_vec)
        #3)如果不够50，补充0。
        for fill0 in range(len(single), 50):
            word_vec = [0.0] * 200
            single_vec.append(word_vec)
            pass
        article_vec.append(single_vec)
    # return article_vec, credData, topic_comm_state, labels
    return article_vec, labels

# # 测试
# article_vec, credData, topic_comm_state, labels = getliar_text_metadata_vectors(dir='../data/liar_train.tsv')
# print("文本向量：")
# print(article_vec)
# print("信用数据")
# print(credData)
# print("主题信息：")
# print(topic_comm_state[1])
# print(topic_comm_state[2])

#
def getColomnToFile(dir):
    file = readFile(dir)
    list_topics = []
    list_speakers = []
    list_community = []
    list_states = []
    list_media = []
    campaign = 0
    interview = 0
    conference = 0
    TV_radio = 0
    socialmedia = 0
    email = 0
    speech = 0
    book = 0
    for line in file:
        i = 0
        for col in line:
            col = col.strip()
            col = col.lower()
            if col == '':
                i += 1
                continue
            # 获得 3主题 的信息， 共143个
            if i == 3:
                col_split = col.split(',')
                for item in col_split:
                    if item not in list_topics:
                        list_topics.append(item)
            # 获得 4发言者 信息 在train中就有2916个，太多了，暂时不考虑
            if i == 4:
                if col not in list_speakers:
                    list_speakers.append(col)
                pass
            # 7党派 信息, 在train中有23个
            if i == 7:
                if col not in list_community:
                    list_community.append(col)
            # 6州 的信息 86个
            if i == 6:
                if col not in list_states:
                    list_states.append(col)
            if i == 13:
                if col not in list_media:
                    list_media.append(col)

                if col.find('speech') != -1:
                    speech += 1
                    continue
                # 统计具体信息
                if col.find('campaign') != -1 or col.find('candidacy') != -1:
                    campaign += 1
                    continue
                if col.find('interview') != -1:
                    interview += 1
                    continue
                if col.find('conference') != -1 or col.find('meeting') != -1 or col.find('hearing') != -1 \
                        or col.find('forum') != -1 or col.find('rally') != -1 or col.find('session') != -1 \
                        or col.find('symposium') != -1:
                    conference += 1
                    continue
                if col.find('tv') != -1 or col.find('radio') != -1:
                    TV_radio += 1
                    continue
                if col.find('social media') != -1 or col.find('article') != -1\
                        or col.find('web') != -1 or col.find('comments') != -1 or col.find('facebook') != -1 \
                        or col.find('twitter') != -1 or col.find('youtube') != -1 or col.find('blog') != -1 \
                        or col.find('tweet') != -1 or col.find('post') != -1 or col.find('.com') != -1 \
                        or col.find('network') != -1 or col.find('internet') != -1 or col.find('online') != -1 \
                        or col.find('instagram') != -1:
                    socialmedia += 1
                    continue
                if col.find('mail') != -1:
                    email += 1
                    continue
            i += 1
    print("speech: %d, campaign: %d, interview: %d, conference: %d, tv_radio: %d, "
          "social_media: %d, email: %d, book: %d"
          % (speech, campaign, interview, conference, TV_radio, socialmedia, email, book))
    # 将不重复的 3主题 list_topics存储到文件中
    wtopicfile = r'../data/statis_3topics.txt'
    with open(wtopicfile, 'w') as f:
        list_topics.sort()
        for elem in list_topics:
            f.write(elem+'\n')
    # 将不重复的 7党派 list_community存储到文件中
    wcommunityfile = r'../data/statis_7community.txt'
    with open(wcommunityfile, 'w') as f:
        list_community.sort()
        for elem in list_community:
            f.write(elem + '\n')
    # 将不重复的 6州 list_community存储到文件中
    wstatefile = r'../data/statis_6state.txt'
    with open(wstatefile, 'w') as f:
        list_states.sort()
        for elem in list_states:
            f.write(elem + '\n')
    mediafile = r'../data/statis_13media.txt'
    with open(mediafile, 'w') as f:
        list_media.sort()
        for elem in list_media:
            f.write(elem + '\n')

# getColomnToFile(dir='../data/liar_train.tsv')


# 获得 6州 信息

# 获得 5发言者的工作标题 信息

