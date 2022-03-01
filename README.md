Необходимые библиотеки:

h5py
hdbscan
jupyter
jupytext
matplotlib
numba
numpy==1.21.3
opencv-python
pandas
Pillow
PyQt==5.14.2
scikit-image==0.17.2
scikit-learn
scipy
swan
tifffile
tqdm
umap-learn

Установка с помощью файла requierments.txt
```
pip install -r requierments.txt
```

Не все библиотеки можно установить напрямую из pip/conda. Для некоторых требуются дополнительные действия.

Napari

```
pip install "napari[all]"
```
Убедитесь, что PyQt5 нужной версии (5.14.2), иначе при запуске napari ноутбук будет падать


Image-funcut

https://github.com/abrazhe/image-funcut/tree/develop
Используем ветку !develop!
В ридми указаны способы установки, но на всякий случай добавлю свой.
- Клонируем репозиторий на свой компьютер
- Переходим в папку с библиотекой.
- Переходим в ветку develop
```
git checkout develop
```
- Запускаем виртуальное окружение (если есть)
- Устанавливаем библиотеку
```
pip3 install -e .
```
- Радуемся!




uCats

https://github.com/abrazhe/uCats
Аналогично предыдущему.
- Клонируем репозиторий на свой компьютер
- Переходим в папку с библиотекой.
- Запускаем виртуальное окружение (если есть)
- Устанавливаем библиотеку
```
pip3 install -e .
```
- Радуемся!



Astromorpho

https://gitlab.com/semyanov-astro/astromorpho
Возможно потребуется доступ так как репозиторий закрытый.
Аналогично предыдущему. Потребуется установленная библиотека image-funcut (см. выше)
- Клонируем репозиторий на свой компьютер
- Переходим в папку с библиотекой.
- Запускаем виртуальное окружение (если есть)
- Устанавливаем библиотеку
```
pip3 install -e .
```
- Радуемся!
