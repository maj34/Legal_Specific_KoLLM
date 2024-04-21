from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import json

results = {}

cnt = 0

for page_number in tqdm(range(1, 11200)):
    
    url = f'https://www.a-ha.io/questions/categories/25?page={page_number}&status=published&order=recent'
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # script 태그에서 JSON 데이터 찾기
        script_tags = soup.find_all('script', type='application/ld+json')
        
        # script_tags가 비어있는지 확인
        if not script_tags:
            continue  # script_tags가 비었다면 다음 반복으로 넘어감
        
        data = json.loads(script_tags[0].string)
        urls = [item['url'] for item in data['mainEntity']['itemListElement']]
        
        if urls:
            for url in urls:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 제목 추출
                    title_tag = soup.find('h1', class_='a-card__headerTitle')
                    title = title_tag.get_text().strip() if title_tag else '제목 없음'
                    
                    # 질문 내용과 답변 내용 추출
                    content_list = []
                    editor_content = soup.find_all('div', 'editor__content')
                    for content in editor_content:
                        content_list.append(content.get_text().strip())
                        
                    if content_list:
                        question = content_list[0]
                        answer = content_list[1:]
                        
                        results[cnt] = {
                            'urls': url,
                            'title': title,
                            'question': question,
                            'answer': answer
                        }
                        
                        cnt += 1
                except Exception as e:
                    print(f"URL 처리 중 오류 발생: {url} - {e}")
        else:
            break
    except Exception as e:
        print(f"페이지 {page_number} 처리 중 오류 발생: {e}")
    
with open('a_ha.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
