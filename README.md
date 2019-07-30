# doggod

## install
1.keras
2.cv2
3.dlib

## start

1. <pre>python datasets.py</pre>
2. revise that
> 1. 파일 형식을 jpg로
3. <pre>python train.py</pre>
> 이미 모델을 넣어놔서 안해도 됨 하지만 정확도 면에서 부족하기 때문에 정확한 데이터를 가지고
>학습시켜야 한다. 
4. <pre>python test.py</pre>
> cv로 실시간으로 디텍팅 해줌 생각보다 성능이 좋다. dlib로 쓰는 것은 너무 느린것에 비해
>거의 초당 10개 정도 처리