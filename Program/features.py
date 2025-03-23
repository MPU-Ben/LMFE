import numpy as np
from collections import Counter
from Bio.Data.CodonTable import unambiguous_rna_by_name
from Bio.Seq import Seq

T = []
t=[]
def gF(X, Y, ss_path, **args):
    T.clear()
    def zCurve(x):
        tmp_t = []
        tmp_x = []
        tmp_y = []
        tmp_z = []
        feature_names = ['x_axis','y_axis','z_axis']
        tmp_t.append(feature_names)
        for sequence in x:
            TU = sequence.count('U')
            A = sequence.count('A'); C = sequence.count('C'); G = sequence.count('G');
            x_ = (A + G) - (C + TU)
            y_ = (A + C) - (G + TU)
            z_ = (A + TU) - (C + G)
            tmp_x.append(x_)
            tmp_y.append(y_)
            tmp_z.append(z_)

        tmp = [[x , y , z] for x, y, z in zip(tmp_x,tmp_y,tmp_z)]
        tmp_t.extend(tmp)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
    def gcContent(x):
        tmp_t = []
        feature_names = ['gcContent']
        tmp_t.append(feature_names)
        for sequence in x:
            TU = sequence.count('U')
            A = sequence.count('A')
            C = sequence.count('C')
            G = sequence.count('G')
            feature_vector = [(G + C) / (A + C + G + TU)  * 100.0]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
    def atgcRatio(x):
        tmp_t = []
        feature_names = ['atgcRatio']
        tmp_t.append(feature_names)
        for sequence in x:
            TU = sequence.count('U')
            A = sequence.count('A')
            C = sequence.count('C')
            G = sequence.count('G')
            feature_vector = [(A + TU) / (G + C)]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
    def NAC(x):
        NA = 'ACGU'
        tmp_t = []
        feature_names = [na for na in NA]
        tmp_t.append(feature_names)
        for sequence in x:
            sequence = sequence.strip()
            count = Counter(sequence)
            feature_vector = []

            for na in NA:
                count_na = count[na] / len(sequence)
                feature_vector.append(count_na)
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T)>0 else tmp_t
    def DNC(x):
        AA = 'ACGU'
        AADict = {}
        code = [0] * 16
        feature_names = []
        feature_values = []
        tmp_t = []

        for i in range(len(code)):
            feature_name = AA[i // 4] + AA[i % 4]
            feature_names.append(feature_name)
        # t.append(feature_names)
        tmp_t.append(feature_names)

        for i in range(len(AA)):
            AADict[AA[i]] = i

        for sequence in x:
            sequence = sequence.strip()
            tmpCode = [0] * 16
            feature_vector = []
            try:
                for j in range(len(sequence) - 2 + 1):
                    tmpCode[AADict[sequence[j]] * 4 + AADict[sequence[j + 1]]] += 1
                if sum(tmpCode) != 0:
                    for i in tmpCode:
                        calcs = i/sum(tmpCode)
                        feature_vector.append(calcs)
                    feature_values.append(feature_vector)

                    tmp_t.append(feature_vector)
            except:
                print(sequence)

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
    def TNC(x):
        AA = 'ACGU'
        AADict = {}
        Code = [0] * 64
        feature_names = []
        feature_values = []
        tmp_t = []

        # 生成特征名
        for i in range(len(Code)):
            feature_name = ''.join([AA[(i >> (2 * j)) & 3] for j in range(2, -1, -1)])#join([AA[j] for j in range(4) if (i >> (2 * j)) & 3])'TNC-' + ''.join()
            feature_names.append(feature_name)
        tmp_t.append(feature_names)

        for i in range(len(AA)):
            AADict[AA[i]] = i

        for sequence in x:
            sequence = sequence.strip()
            tmpCode = [0] * 64
            feature_vector = []
            try:
                for j in range(len(sequence) - 3 + 1):
                    tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j + 1]] * 4 + AADict[sequence[j + 2]]] += 1
                if sum(tmpCode) != 0:
                    for i in tmpCode:
                        calcs = i / sum(tmpCode)
                        feature_vector.append(calcs)
                    feature_values.append(feature_vector)
                    # t.append(feature_vector)
                    tmp_t.append(feature_vector)
                    # trackingFeatures.append(feature_names)
            except:
                print(sequence)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

        # print()
    def SeqLength(x):
        tmp_t = []
        feature_name = ['SeqLength']
        tmp_t.append(feature_name)
        for sequence in x:
            seq = Seq(sequence)
            feature_vector = [len(seq)]
            tmp_t.append(feature_vector)

        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        print()
    def orf_coverage(x):
        tmp_t = []
        feature_names = ['ORFs_Coverage']
        tmp_t.append(feature_names)
        codon_table = unambiguous_rna_by_name["Standard"]
        start_codons = codon_table.start_codons  # ["AUG"]
        stop_codons = codon_table.stop_codons  # ["UAA", "UAG", "UGA"]

        for sequence in x:
            # 转换为RNA序列
            # rna_sequence = sequence.replace('T', 'U')
            seq = Seq(sequence)
            # 计算正向链上的ORF覆盖率
            forward_coverage = 0
            for i in range(len(seq) - 2):
                codon = str(seq[i:i + 3])
                if codon in start_codons:
                    j = i + 3
                    while j < len(seq) - 2:
                        codon = str(seq[j:j + 3])
                        if codon in stop_codons:
                            forward_coverage += j - i + 3
                            break
                        j += 3
            orf_coverage = forward_coverage
            feature_vector = [orf_coverage  / (len(seq))]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

    def orf_length(x):
        # 初始化结果列表
        tmp_t = []
        feature_names = ['ORFs_Length']
        tmp_t.append(feature_names)
        # 使用通用 RNA 遗传密码表
        codon_table = unambiguous_rna_by_name["Standard"]
        start_codons = codon_table.start_codons  # ["AUG"]
        stop_codons = codon_table.stop_codons  # ["UAA", "UAG", "UGA"]
        # 遍历每个序列
        for sequence in x:
            seq = Seq(sequence)
            # 初始化 ORF 总长度
            total_orf_length = 0
            current_length = 0
            in_orf = False

            # 遍历序列以查找 ORF
            for i in range(0, len(seq) - 2, 3):
                codon = str(seq[i:i + 3])

                if not in_orf and codon in start_codons:
                    in_orf = True
                    current_length = 3  # 起始密码子长度
                elif in_orf:
                    if codon in stop_codons:
                        in_orf = False
                        total_orf_length += current_length  # 将当前 ORF 长度加入总长度
                        current_length = 0
                    else:
                        current_length += 3  # 增加长度

            # 检查最后一个 ORF
            if in_orf and current_length > 0:
                total_orf_length += current_length  # 将最后一个 ORF 的长度加入总长度

            # 将总 ORF 长度添加到结果中
            tmp_t.append([total_orf_length])

        # 更新全局 T
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

    def orf_count(x):
        tmp_t = []
        feature_names = ['ORFs_Count']
        tmp_t.append(feature_names)
        codon_table = unambiguous_rna_by_name["Standard"]
        start_codons = codon_table.start_codons  # ["AUG"]
        stop_codons = codon_table.stop_codons  # ["UAA", "UAG", "UGA"]

        for sequence in x:
            orf_count = 0
            seq = Seq(sequence)
            # 搜索正向链
            for i in range(len(seq) - 2):
                codon = str(seq[i:i + 3])
                if codon in start_codons:
                    j = i + 3
                    while j < len(seq) - 2:
                        codon = str(seq[j:j + 3])
                        if codon in stop_codons:
                            orf_count += 1
                            break
                        j += 3

            feature_vector = [orf_count]
            tmp_t.append(feature_vector)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
    def getMFE():
        import secondaryFeatures
        tmp_t = []
        feature_names = ['MFE','Num.Base_pairs','num_AU_pairs','num_GC_pairs','num_internal_loops', 'num_external_loops','num_unpaired_bases']
        tmp_t.append(feature_names)
        mfelist = secondaryFeatures.MFE_Feat(ss_path)
        tmp_t.extend(mfelist)
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t
        # print()
    def getFeatures(x,y):

        if args['SeqLength'] == 1:
            SeqLength(x)

        if args['zCurve'] == 1:
            zCurve(x)

        if args['gcContent'] == 1:
            gcContent(x)

        if args['atgcRatio'] == 1:
            atgcRatio(x)

        if args['NAC']==1:
            NAC(x)

        if args['DNC']==1:
            DNC(x)

        if args['TNC']==1:
            TNC(x)

        if args['orf_length'] == 1:
            orf_length(x)

        if args['orf_coverage'] == 1:
            orf_coverage(x)

        if args['count_orfs'] == 1:
            orf_count(x)

        if args['MFE'] == 1:
            getMFE()

        tmp_t = []
        feature_name = ['label']
        tmp_t.append(feature_name)
        for i in range(len(y)):
            tmp_t.append([y[i]])
        global T
        T = [row_a + row_b for row_a, row_b in zip(T, tmp_t)] if len(T) > 0 else tmp_t

    getFeatures(X, Y)
    return np.array(T)

