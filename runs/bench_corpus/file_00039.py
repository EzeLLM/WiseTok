def commonCharacterCount(s1,s2):
    s1_list = []
    s2_list = []
    for x in s1:
        s1_list.append(x)
    
    for x in s2:
        s2_list.append(x)
    
    common = list(set(s1_list).intersection(s2_list))
    
commonCharacterCount('abbbbs', 'seeab')