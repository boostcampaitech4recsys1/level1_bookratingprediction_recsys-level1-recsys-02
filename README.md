# Book Recommendataion

사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 프로젝트이다. 책 한권을 모두 읽기 위해서는 보다 긴 물리적인 시간을 필요로 한다. 또한 소비자 입장에서는 제목, 저자, 표지, 카테고리 등 한정된 정보로 각자가 콘텐츠를 유추하고 구매 유무를 결정해야 하기 때문에 상대적으로 선택에 더욱 신중을 가하게 된다.

이러한 소비자들의 책 구매 결정에 대한 도움을 주고자 한다. 사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 모델을 구축 한다.

책에 대한 메타 데이터인 books, 고객에 대한 메타 데이터인 users, 고객이 책에 남긴 평점 ratings 의 데이터 셋을 활용해, 최종적으로 1과 10 사이 평점을 예측하는 것을 목적으로 한다.

(데이터셋 출처: https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset)

# Project architecture

```
├─src
	├─data
        ├─context_data.py
        ├─ ~~ 이하 생략 ~~
        ├─text_data.py
	├─ensembles
	├─models
        ├─dl_models.py
        ├─
        ├─
├─main.py
├─ensemble.py
├─requirements.txt
```

# Environment Requirements

```
catboost==1.1.1
fasttext==0.9.2
nltk==3.7
numpy==1.23.4
pandas==1.5.1
Pillow==9.3.0
scikit_learn==1.1.3
torch==1.7.1
torchvision==0.8.2
tqdm==4.51.0
transformers==4.23.1
```

# Example to Run the Codes

# Reference

- [CatBoost](https://github.com/catboost/catboost)
- [Factorization Machine](https://ieeexplore.ieee.org/document/5694074)
- [Field-aware Factorization Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Neural Collaborotaive Filtering]("https://arxiv.org/abs/1708.05031")
- [DeepCoNN](https://arxiv.org/abs/1701.04783)

# Contributors

| <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/70f87803-443d-4831-9442-7c46e477ea6c/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221106%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221106T071629Z&X-Amz-Expires=86400&X-Amz-Signature=2a4befb0a65968d4d129bed6010e5c2e5e1ef57ebd5eec9e9912f85201f2c959&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22image.png%22&x-id=GetObject" width=200> | <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c08108e8-fae6-424b-978a-06b1e64692ac/KakaoTalk_20221002_232533360.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221106%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221106T071934Z&X-Amz-Expires=86400&X-Amz-Signature=7f70f512814d29391762369197456a92f06563b7ba4a90a7817a03fe026a1a1b&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22KakaoTalk_20221002_232533360.jpg%22&x-id=GetObject" width=200> | <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b63ad532-9183-41a2-b7f4-ab3e7ab227af/KakaoTalk_20221001_013107840.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221106%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221106T072004Z&X-Amz-Expires=86400&X-Amz-Signature=efb5cb972aea798fb45aa545b25e24ed6bf6d34de13afe28297ad999d83fa517&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22KakaoTalk_20221001_013107840.jpg%22&x-id=GetObject" width=200> | <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9ef73f08-943c-449e-bf59-308fc4900f10/i_ff8605ba4220.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221106%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221106T071800Z&X-Amz-Expires=86400&X-Amz-Signature=c0b131c15656354416fbbe4e14863a56ae69922c189c1ff447a365fa7de1d3bf&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22i_ff8605ba4220.jpg%22&x-id=GetObject" width=200> | <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8918b865-4e35-46f7-b5cb-b740e6e3db7d/KakaoTalk_20220919_192231266.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221106%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221106T072025Z&X-Amz-Expires=86400&X-Amz-Signature=f2950080f5ef415a2dfef74bbd0c276ce7cc955b2ee06bd45011937170d79f3a&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22KakaoTalk_20220919_192231266.jpg%22&x-id=GetObject" width=200> |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                                                                                                                                                                          [류명현](https://github.com/ryubright)                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                 [이수경](https://github.com/41ow1ives)                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                  [김은혜](https://github.com/kimeunh3)                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                 [정준환](https://github.com/Jeong-Junhwan)                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                  [장원준](https://github.com/jwj51720)                                                                                                                                                                                                                                                                  |
