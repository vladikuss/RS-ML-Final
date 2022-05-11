В проекте использан [этот датасет](https://www.kaggle.com/competitions/forest-cover-type-prediction).

Добавлены требования в формате YML.

Проект включает следующие скрипты:
* подготовка данных - preprocess.py
* моделирование данных -  forest.py, model.py
* сохранение данных - saving.py

*** 

Чтобы сохранить модель используем команду:
```sh
python saving.py save
```

* Метрики на Comet ML:

![image](https://user-images.githubusercontent.com/61574055/167886471-804f1d5f-2474-450d-ab8a-752e2f82e18f.png)

![image](https://user-images.githubusercontent.com/61574055/167886653-5a9d6b49-b729-4c1e-bf6d-ca252bc9fdfa.png)

* Результаты на kaggle:

![image](https://user-images.githubusercontent.com/61574055/167849506-344ae7d8-a981-48e9-b6eb-15733427912e.png)
