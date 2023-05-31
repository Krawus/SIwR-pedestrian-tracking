# Sztuczna Inteligencja w Robotyce - projekt zaliczeniowy
## Opis projektu
Projekt polega na stworzeniu systemu śledzącego przechodniów wykorzystującego probablistyczne modele grafowe. System ma za zadanie określić położenie przechodniów na kolejnych klatkach obrazu z kamery poprzez przypisane prostokątów ograniczających (ang. Bounding Boxes, BBoxes) do poszczególnych osób.
Głównym celem jest określenie którym BoundingBox z poprzedniej klatki odpowiadają BoundingBox z aktualnej klatki.
Każdy BoundingBox posiada przypisaną liczbęm, oznaczającą indeks BoundingBoxa z poprzedniej klatki -> przyjmuje opdowiednio -1 w przypadku gdy np. przechodzień dopiero pojawia się na ekranie.
## Metodyka
Główną strukturą pozwalającą na realizacje zadania jest *Bipartite Graph* (graf dwudzielny), którego węzły dzielimy na dwa rozłączne zbiory - odpowiednio BoundingBoxy z klatki poprzedniej i BoundingBoxy z aktualnej klatki kamery - w taki sposób aby krawędzie nie łączyły węzłów nalezących do tego samego zbioru.
<p align="center" width="70%">
    <img width="33%" src=https://github.com/Krawus/SIwR-pedestrian-tracking/blob/main/readmeFiles/graph.png?raw=true> 
</p>

Graf ten pozwala na obliczenie prawdopodobieństwo, iż odpowiedni BoundingBox z klatki *n* odpowiada temu samemu obiektowi co odpowiedni BoundingBox z klatki *n-1*. Zatem koszty przejścia w przedstawionym grafie reprezentują prawdopodobieństwo opisywanych powyżej zdarzeń.    
W analizowanym przypadku każdy z BoundingBoxów pojawiających się w klatce *n* może reprezentować nowego przechodnia - w związku z czym aby uwzględnić ten przypadek należy odpowiednio rozbudować graf.
<p align="center" width="70%">
    <img width="33%" src=https://github.com/Krawus/SIwR-pedestrian-tracking/blob/main/readmeFiles/graph2.png> 
</p>







