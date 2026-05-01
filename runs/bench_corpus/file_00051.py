# 전화번호 목록
'''
    검색이 빈번 -> 해쉬(dict)

    접근
    1) 주어진 리스트 내의 문자열, 길이를 쌍으로 구성
    2) 문자열 길이가 낮은 순으로 정렬(=> 슬라이싱 에러 방지 ex. 길이가 3인데 5자리로 슬라이싱하면 에러)
    3) 길이가 짧은 순으로 기준잡아서 기준 잡은 값 다음부터 슬라이싱
    4) 배열에 담아서 있는지 확인
    5) 효율성 테스트에서 2개 시간초과.. O(n^2) 이라 그런듯..?

    import operator
    def solution(phone_book):
        answer = True # 접두어가 없는 경우
        saveDict = dict()

        # 문자열, 문자열 길이로 구성
        for i in phone_book:
            saveDict[i] = len(i)

        # [('119', 3), ('97674223', 8), ('1195524421', 10)]
        # operator => value 값으로 오름차순 정렬하기 위해
        sortDict = sorted(saveDict.items(), key=operator.itemgetter(1))
        for idx, val in enumerate(sortDict):
            temp = []
            for j in sortDict[idx+1:]:
                temp.append(j[0][0:val[1]])

            if val[0] in temp:
                return False
        return answer
'''
''' 
    다른사람풀이..

    operator 없이 길이로 정렬
     - phone_book.sort(key=lambda x: len(x))
     - phone_book = sorted(phone_book, key= len)

    [ 해시 이용 ]
     def solution(phone_book):
        answer = True
        dic ={} #key,value형태의 딕셔너리이용
        for pNumber in phone_book:
            dic[pNumber] = 1 #key:폰번호 value:1
        for pNumber in phone_book: #각각 폰번호마다 검사
            temp=""
            for num in pNumber: #폰번호를 한글자로 쪼개서 반복문 "243"이면 "2" "4" "3"
                temp +=num #쪼갠 숫자를 반복문이 돌아갈 때마다 붙음  
                if temp in dic and temp!=pNumber: #딕셔녀리의 키로 존재하는지 검사
                    answer = False
        return answer

    [ sort ]
    def solution(phone_book):
        phone_book.sort(key=lambda x: len(x)) #길이로 정렬 
    print(phone_book)
    for a in range(len(phone_book)): #리스트 기준요소를 반복문 돌리기
        for b in range(a+1, len(phone_book)): #기준 요소의 다음 요소부터 반복문 돌리기
            if phone_book[b][:len(phone_book[a])] == phone_book[a]: #다음 요소가 기준요소의 길이로 잘랐을때 값이 같으면 false
                return False
    return True
'''
# 다른사람풀이(2) - zip이용
'''
  Q. ['119', '1195524421', '97674223'] 일때 119가 n번째에서 나올 수도 있는데 
  zip쓰면 첫번째 끼리 묶이니까 탐색안하는거 아님?
  A. 그래서 정렬함. 작은수~큰수가 아니라 문자열이기 때문에 
  각 자리마다 1~9 오름차순으로 정렬됨 이때 자리수가 적으면 먼저나오겠지..접두어로 쓰여질 확률도 높고
  따라서 zip(phone_book, phone_book[1:]) 써도 무관 + 다음꺼랑 비교를 위해서 사용
'''
def solution(phone_book):
    # 문자열이라 큰 숫자대로 정렬되는게 아님 (1~9순서)
    phone_book = sorted(phone_book) 
    for p1, p2 in zip(phone_book, phone_book[1:]):
        if p2.startswith(p1): #startswith()함수 배워갑니다...
            return False
    return True

# print(solution(["97674223", "1195524421", "119"]))
# print(solution(["123","456","789"]))

# for pair in zip(['119', '1195524421', '97674223'], ['1195524421', '97674223']):
    # print(pair)
# (119,1195524421), (1195524421,97674223)