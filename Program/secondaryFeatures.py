import re
from numpy.core.defchararray import upper
def extract_secondary_structure_features(dot_bracket_notation,seq_notation):
    stack = []
    base_pairs = []
    external_loops = []
    internal_loops = []
    unpaired_bases = 0
    longest_external_loop = -1
    longest_internal_loop = -1

    for i, symbol in enumerate(dot_bracket_notation):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')':
            if len(stack) > 0:
                j = stack.pop()
                base_pairs.append((j, i))
            else:
                raise ValueError("Unmatched closing bracket")
        elif symbol == '.':
            unpaired_bases += 1
            if len(stack) == 0:
                external_loops.append(i)
                longest_internal_loop = max(longest_internal_loop, len(internal_loops))
            else:
                internal_loops.append(i)
                longest_internal_loop = max(longest_internal_loop, len(internal_loops))
        else:
            raise ValueError(f"Invalid symbol: {symbol}")

    if len(stack) > 0:
        raise ValueError("Unmatched opening bracket")

    if len(internal_loops) > 0:
        longest_internal_loop = max(longest_internal_loop, len(internal_loops))

    longest_external_loop = max(longest_external_loop, len(external_loops))
    # print(dot_bracket_notation)
    num_base_pairs = len(base_pairs)
    try:
        num_AU_pairs = sum(1 for i, j in base_pairs if seq_notation[i] == 'A' and seq_notation[j] == 'U' and (i, j) in base_pairs)
        num_GC_pairs = sum(1 for i, j in base_pairs if seq_notation[i] == 'G' and seq_notation[j] == 'C' and (i, j) in base_pairs)
        num_circles = dot_bracket_notation.count('()')
        num_internal_loops = len(internal_loops)
        num_external_loops = len(external_loops)
        length_long_internal_loop = longest_internal_loop
        length_long_external_loop = longest_external_loop
        num_unpaired_bases = unpaired_bases
    except:
        print(dot_bracket_notation)
        print(seq_notation[i])
    struct_features = {
        'num_base_pairs': num_base_pairs,
        'num_AU_pairs': num_AU_pairs,
        'num_GC_pairs': num_GC_pairs,
        'num_internal_loops': num_internal_loops,
        'num_external_loops': num_external_loops,
        'num_unpaired_bases': num_unpaired_bases
    }
    return struct_features

def MFE_Feat(specie_ss_path):
    sequences = []
    feature_vectors = []
    file_path = specie_ss_path
    with open(file_path, 'r') as file:
        lines = file.readlines()
        sequence = ''
        for line in lines:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                sequence = ''
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    # return sequences
    for seq in sequences:
        try:
            # pattern_des = r"^>(.+)$"
            feature_vector = []
            pattern_mfe = r"\(([-+]?\d+\.\d+)\)"
            pattern_seq = r"([ACGU]+)"
            pattern_struct = r"\s*([.()]+)\s*"

            match_mfe = re.search(pattern_mfe, str(upper(seq)))
            match_len = re.search(pattern_seq, str(upper(seq)))
            match_struc = re.search(pattern_struct, str(upper(seq)))

            if match_mfe and match_len and match_struc:
                min_free_energy = float(match_mfe.group(1))
                length= len(match_len.group(1))
                try:
                    normal = float(min_free_energy / (length))
                    feature_vector.append(normal)
                    feature_vector.extend(extract_secondary_structure_features(match_struc.group(1),match_len.group(1)).values())
                except ZeroDivisionError:
                    print('ZeroDivisionError',seq)
            else:
                print(seq)
                feature_vector.append('misMatch:')

        except IOError as e:
            print('IO ERROR!', e)
        feature_vectors.append(feature_vector)
    return feature_vectors



if __name__ == '__main__':
    MFE_Feat()