# Sztuczna Inteligencja w Robotyce - projekt zaliczeniowy
## Opis projektu
Projekt polega na stworzeniu systemu śledzącego przechodniów wykorzystującego probablistyczne modele grafowe. System ma za zadanie określić położenie przechodniów na kolejnych klatkach obrazu z kamery poprzez przypisane prostokątów ograniczających (ang. Bounding Boxes, BBoxes) do poszczególnych osób.
Głównym celem jest określenie którym BoundingBox z poprzedniej klatki odpowiadają BoundingBox z aktualnej klatki.
Każdy BoundingBox posiada przypisaną liczbęm, oznaczającą indeks BoundingBoxa z poprzedniej klatki -> przyjmuje opdowiednio -1 w przypadku gdy np. przechodzień dopiero pojawia się na ekranie.
## Metodyka
Główną strukturą pozwalającą na realizacje zadania jest *Bipartite Graph* (graf dwudzielny), którego węzły dzielimy na dwa rozłączne zbiory - odpowiednio BoundingBoxy z klatki poprzedniej i BoundingBoxy z aktualnej klatki kamery - w taki sposób aby krawędzie nie łączyły węzłów nalezących do tego samego zbioru. Przedstawiony poniżej został przypadek w którym zarówno w poprzedniej jak i aktualnej klatce zostały wykryte po 3 BoundingBoxy.  
<p align="center">
    <img src=https://github.com/Krawus/SIwR-pedestrian-tracking/blob/main/readmeFiles/graph.png> 
</p>

Graf ten pozwala na obliczenie prawdopodobieństwo, iż odpowiedni BoundingBox z klatki *n* odpowiada temu samemu obiektowi co odpowiedni BoundingBox z klatki *n-1*. Zatem koszty przejścia w przedstawionym grafie reprezentują prawdopodobieństwo opisywanych powyżej zdarzeń.    
W analizowanym przypadku każdy z BoundingBoxów pojawiających się w klatce *n* może reprezentować nowego przechodnia - w związku z czym aby uwzględnić ten przypadek należy odpowiednio rozbudować graf.
<p align="center">
    <img src=https://github.com/Krawus/SIwR-pedestrian-tracking/blob/main/readmeFiles/graph2.png> 
</p>

Tak zdefiniowany graf pozwala na zdefiniowanie tabeli, w której wiersz reprezentuje indeks BoundingBoxa na aktualnej klatce *n*-tej, natomiast kolumna reprezentuje indeks BoundingBoxa z klatki poprzetniej *n-1*-tej. Każdy BoundingBox posiada jednakowe prawdopodobieństwo pojawienia się na ekranie po raz pierwszy.
(poniższa tabela została uzupełniona przykładowymi danymi)
<center>

|   |  0  |   1  |   2  | new | new | new |
|---|:---:|:----:|:----:|:---:|:---:|-----|
| 0 | 0.9 | 0.2  | 0.22 | 0.4 | 0.4 | 0.4 |
| 1 | 0.8 | 0.5  | 0.32 | 0.4 | 0.4 | 0.4 |
| 2 | 0.7 | 0.65 | 0.17 | 0.4 | 0.4 | 0.4 |

</center>

Tak skonstruowana tabela pozwala odczytac przykładowo prawdopodobieństwo iż przechodzień znajdujący się aktualnie w BoundingBoxie o indeksie 0 jest tą samą osobą, która została wykryta w BoundingBoxie o indeksie  1 na poprzedniej klatce (w tym przypadku prawdopodobieństwo wynosi 0.2).  
Aby dokonac prawidłowego przypisania indeksów aktualnych BoundingBoxów należy dla każdego obiektu znaleźć największe prawdopodobieństwo.  
Aby nie doszło do dwukrotnego przypisania tego samego indeksu danemu boundingBoxowi oraz aby rozwiązanie nie było dobierane w sposób zachłanny wykorzystano metrykę, znajdującą optymalne rozwiązanie gwarantujące największą sumę prawdopodobieństw BoundingBoxów - *Hungarian algorithm https://en.wikipedia.org/wiki/Hungarian_algorithm*.
  

## Prezentowane rozwiązanie
W prezentowanym rozwiązaniu zastępuje wczytanie danych do analizy. Na początku wszystkim znalezionym na pierwszej klatce BoundingBoxom przypisane zostają indeksy -1. Następnie na kolejnych parach sąsiadujących ze sobą klatek dokonywane jest przetwarzanie. Wycianane i przechowywane są fragmenty obrazu odpowiadające kolejnym BoundingBoxom, a następnie konwertowane do przestrzeni barw HSV - na podstawie odpowiednich kanałów przesteni HSV obrazy będą ze sobą porównywane. Następnie następuje przycięcie obrazów reprezentujących BoundingBoxy o odpowiedni ułamek w osi X i osi Y - aby nie brać pod uwagę tła i zbędnych elementów otoczenia, a jedynie bardziej charakterystyczną przestrzeń BoundingBoxa.  
Kolejno zostaje utworzona macierz reprezentująca kolejne przejcia w grafie *Bipartite Graph*. Macierz ta jest o wymiarze *(liczba BoundingBox w klatce aktualnej x liczba BoundingBox w klatce poprzedniej + liczbaBouningBox w klatce aktualnej)*, ponieważ każdy przechodzień z aktualnej klatki może pojawić się po raz pierwszy. Macierz ta uzupełniana jest odpowiednimi wartościami reprezentującymi podobieństwo odpowiadających sobie obrazów w BoundingBoxach. Jako metrykę podobieństwa przyjęto korelację pomiędzy histogramami poszczególnych obrazów w przestrzeni H(hue) oraz S(saturation).
Finalnie odpowiednie indeksy zostały przypisane BoundingBoxom na podstawie tabeli, dzieki zastosowaniu opisywanego wcześniej *Hungarian algorithm*. W przypadku gry prawdopodobieństwo przypisania danego indeksu odpowiadało identyfikacji BoundingBoxa jako nowego - przypisywano wartość -1.


### Uruchomianie
``` 
python3 main.py <pathToData>
```