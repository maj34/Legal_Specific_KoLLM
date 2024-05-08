from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import json
import re
import html

# 세션 시작
session = requests.Session()

law_article = {}

for page_num in tqdm(range(0, 9735)):
    idx = (page_num - 1) * 10
    
    url = f'https://www.lawtimes.co.kr/Search?q=&type=%EC%A0%84%EC%B2%B4&category=news&page={page_num}'
    response = session.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 페이지 내의 URL 찾기
    href_tags = soup.find_all('a', class_='css-5umq2t ettzvl60')
    for tag in href_tags:
        try:
            sub_article_url = 'https://www.lawtimes.co.kr' + tag['href']
            sub_article_response = session.get(sub_article_url)
            sub_article_soup = BeautifulSoup(sub_article_response.text, 'html.parser')

            # 기사 제목
            head = sub_article_soup.find('meta', property='og:title')['content']
            
            # 기사 본문
            if meta := sub_article_soup.find('meta', attrs={"name": "description"}):
                description_content = meta['content']
                cleaned_content = BeautifulSoup(description_content, 'html.parser').get_text()
                cleaned_content = html.unescape(cleaned_content)
                cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
                cleaned_content = re.sub(r'\\+', '', cleaned_content).strip()
                cleaned_content = re.sub(r'\'', '', cleaned_content)
                
                law_article[idx] = {
                    'title': head,
                    'content': cleaned_content,
                    'url': sub_article_url
                }
                idx += 1
        except Exception as e:
            print(e)
            continue

with open('law_articles.json', 'w', encoding='utf-8') as f:
    json.dump(law_article, f, ensure_ascii=False, indent=4)