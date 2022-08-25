# Tirads bounding box prediction with Yolov5

Запуск:

-- До начала запуска клонируйте данный репозиторий `git clone https://github.com/ksunchikk/tirads`

-- Затем подготовьте среду для работы: установите модуль torch и выполните в командной строке `pip install -r requirements.txt` (из дериктории клонированного репозитория)

-- Для предсказания бокса файла с разрешением .tif необходимо запустить на исполнение скрипт `detect_tif.py`

-- Исполнение данного файла возможно при указании 3 параметров:



     -s (--source_path) - путь исходного файла для детекции
     -o (--out_path) - путь файла, куда необходимо сохранить результат
     -m (--model_path) - путь для модели, осуществляющей детекцию
     
 -- Все обученные модели лежат в директории `weights` данного репозитория



     -3, -5, -10, -15, -all в названии файла означают количество изображения, используемых для обучения модели, сохраненной в .pt файле (каждое третье изображение, каждое пятое и т.д)
     -both, -cross, -long - в названии файла означают вид изображений, используемых для обучения модели, сохраненной в .pt файле (все изображения, только продольные, только поперечные, соответственно)
  -- Для начала исполнения в командной строке выполните 
  
    python detect_tif.py -s "C:/Users/example/example.tif" -o "C:/Users/example/result.tif" -m "weights/3_both_model.pt" 
     
     
     
