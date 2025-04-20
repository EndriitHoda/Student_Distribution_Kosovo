# Universiteti i Prishtinës "Hasan Prishtina"

<div align="center">
  <img src="uni-pr.png" alt="University Logo" title="University Logo" width="200">
</div>

<div align="center">
<b>Fakulteti: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</b><br>
<b>Departamenti: Departamenti i Inxhinierisë Kompjuterike</b><br>
<b>Lënda: Machine Learning</b><br>
<b>Niveli: Master</b><br>
<b>Profesori: Prof. Dr. Ing. Lule Ahmedi</b><br>
<b>Asistenti: Dr. Sc. Mërgim Hoti</b><br>
</div>
<br>


# Machine Learning: Optimizing School Resource Allocation

Ky repository përdoret për qëllime studimore në fushën e Machine Learning, për një analizë mbi arsimin ne Kosove duke perfshire nxenesat, stafin akademik dhe stafin administrativ ne komunat e Republikes se Kosoves.

    ## Pjesëmarrësit në Projekt

Studentët që kanë marrë pjesë në këtë projekt janë:
- Endrit Hoda
- Lorik Mustafa
- Meriton Kryeziu

## Burimi i Dataseteve

Datasetet që do të përdoren janë marrë nga ASK (Agjencia e Statistikave të Kosovës) dhe përmbajnë të dhëna nga e njëjta periudhë kohore. Këto datasetë kanë korrelacion dhe ndikojnë njëri-tjetrin, prandaj kemi vendosur t'i përdorim si burime për këtë projekt.

# Faza e pare: Përgatitja e modelit
## Nxjerrja e Dataseteve nga ASK
Përmes script-ave `DataScraper.py` dhe `Scraper.py`, datasetet e mëposhtme janë marrë nga ASK:
```python

datasets = {
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/1 Pupils in public education/edu05.px": "dataset.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/2 Educational staff/edu16.px": "dataset_stafi_akademik.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/2 Educational staff/edu19.px": "dataset_stafi_administrativ.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/2 Educational staff/edu21.px": "dataset_stafi_ndihmes.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/5 Learning conditions/edu26.px": "dataset_raportet_nxenes_staf.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Education/6 Information on social and family status of pupils/edu28.px": "dataset_statusi_social.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Household budget survey/1 Overall consumption in Kosovo/hbs01.px": "dataset_konsumi_i_pergjithshem.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Environment/Uji/tab14.px": "dataset_uji_humbur.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Population/Births and deaths/birthsDeathsHistory.px": "dataset_lindjet_vdekjet.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/Population/Estimate, projection and structure of population/Population projection/Population projection, 2011-2061/tab1vMesem.px": "dataset_populsia_with_prediction.csv",
    "https://askdata.rks-gov.net/api/v1/sq/ASKdata/ICT/ICT in Households/TIK1point1.px": "dataset_qasja_ne_internet.csv"
}

for url, csv_name in datasets.items():
    fetch_api_output_to_csv(url, csv_name)
```

## Përshkrimi i Dataseteve

Dataseti kryesor përmban gjithsej 2,449 rekorde me 7 atribute. Datasetet relevante kontekstuale kanë një numër të ndryshëm rekordesh dhe atributesh, por gjithmonë përfshijnë periudhën kohore, komunat dhe vitet e regjistruara.

Përshkrimi i datasetet dhe tipet e të dhënave:
- **Numri i nxënësve për komuna** 
```
Komuna              object
Viti Akademik       object
Niveli Akademik     object
Mosha               object
Gjinia              object
Numri i nxenesve    object
dtype: object
       Komuna Viti Akademik Niveli Akademik Mosha    Gjinia Numri i nxenesve
count    2448          2448            2448  2448      2448             2448
unique     34             9               4     4         2             1663
top     Deqan     2015-2016     Para-fillor   5-6  Mashkull                -
freq       72           272             612   612      1224              101
```

- **Stafi akademik**
```
viti                            object
komuna                          object
variabla                        object
gjinia                          object
Mësimdhënësit sipas niveleve    object
dtype: object
             viti komuna      variabla   gjinia Mësimdhënësit sipas niveleve
count        2835   2835          2835     2835                         2835
unique          9     35             3        3                          667
top     2015/2016  Deçan  Parashkollor  Meshkuj                            -
freq          315     81           945      945                          590
```

- **Stafi administrativ**
```
viti                                      object
komuna                                    object
personeli administrativ                   object
gjinia                                    object
Personeli administrativ sipas niveleve    object
dtype: object
             viti komuna personeli administrativ   gjinia  \
count        2835   2835                    2835     2835   
unique          9     35                       3        3   
top     2015/2016  Deçan            Parashkollor  Meshkuj   
freq          315     81                     945      945
```

- **Statusi social**
```
viti                                      object
niveli i arsimit                          object
variabla                                  object
gjinia                                    object
Statusi social dhe familjar i nxënësve     int64
dtype: object
       Statusi social dhe familjar i nxënësve
count                              288.000000
mean                              2604.708333
std                               4468.647824
min                                  0.000000
25%                                 88.750000
50%                                694.000000
75%                               2840.250000
max                              30264.000000
```

- **Raportet nxënës mësim-dhënës**
```
viti                                                    object
niveli i arsimit                                        object
treguesi                                                object
Treguesit e arsimit sipas niveleve në arsimin publik    object
dtype: object
             viti     niveli i arsimit          treguesi  \
count         144                  144               144   
unique          9                    4                 4   
top     2015/2016  Arsimi parashkollor  Numri i nxënësve   
freq           16                   36                36   

       Treguesit e arsimit sipas niveleve në arsimin publik  
count                                                 144    
unique                                                 98    
top                                                  0.93    
freq                                                    9   
```

- **Populsia dhe parashikimi i saj**
```
grup- mosha                                                          object
viti                                                                  int64
gjithsej/meshkuj/femra                                               object
Popullsia sipas grup moshës, viteve dhe gjinisë (Variant i mesëm)     int64
dtype: object
              viti  \
count   627.000000   
mean   2036.000000   
std      15.824012   
min    2011.000000   
25%    2021.000000   
50%    2036.000000   
75%    2051.000000   
max    2061.000000  
```

- **Lindjet/Vdekjet**
```
Viti                    int64
Variablat              object
Lindjet dhe vdekjet    object
dtype: object
              Viti
count   150.000000
mean   1985.000000
std      21.721236
min    1948.000000
25%    1966.250000
50%    1985.000000
75%    2003.750000
max    2022.000000
```

## Paraprocesimi dataseteve

Rekordet në datasetet e marra nga regjistra të ndryshëm të ASK kishin vlera të ndryshme për të njëjtin kuptim, prandaj ato janë unifikuar në një format të vetëm. Gjithashtu, vlerat e munguara janë plotësuar me vlera të ndërmjetme, duke ruajtur trendin e rritjes ose uljes së vlerave numerike që mungonin. Një shembull i këtij procesi është riemërtimi dhe plotësimi i vlerave null në datasetin "Raportet nxënës-mësimdhënës".
```python
nxenes_staf_df = pd.read_csv(r'./datasets/dataset_raportet_nxenes_staf.csv')
nxenes_staf_df.rename(columns={'Treguesit e arsimit sipas niveleve në arsimin publik':'vlera'}, inplace=True)
nxenes_staf_df.replace(":", None, inplace=True)
nxenes_staf_df.replace("-", None, inplace=True)

nxenes_staf_df["viti"] = nxenes_staf_df["viti"].str.split("/").str[0]

nxenes_staf_df["vlera"] = pd.to_numeric(nxenes_staf_df["vlera"], errors="coerce")

nxenes_staf_df.sort_values(by=["niveli i arsimit", "viti"], inplace=True)

nxenes_staf_df["vlera"] = nxenes_staf_df.groupby("treguesi")["vlera"].transform(lambda group: group.interpolate(method="linear")).round(1)

nxenes_staf_df.to_csv("./cleaned_datasets/dataset_raportet_nxenes_staf.csv", index=False)
```

# Faza e dytë: Trajnimi e modelit
Gjatë kësaj faze, grupi ynë ka vendosur të krijojë dhe të trajnojë modele të ndryshme të Machine Learning (ML) mbi datasetin që përshkruan shpërndarjen e studentëve në Kosovë. Qëllimet tona kryesore janë:

- Parashikimi i numrit të studentëve në qytete të ndryshme të Kosovës për pesë vitet e ardhshme.

- Gjenerimi i klasterëve të ndryshëm për të analizuar shpërndarjen aktuale të studentëve.

- Vizualizimi i shkallëzueshmërisë së numrit të studentëve në komuna të ndryshme përgjatë viteve të kaluara dhe për të ardhmen.

## Modelet dhe Algoritmet e Aplikuara
Në total janë trajnuar 5 modele me algoritme të ndryshme të ML, prej të cilave:

- 3 Supervised: Random Forest, XGBoost, ARIMA

- 2 Unsupervised: K-Means Clustering, DBSCAN + t-SNE

### 1. Random Forest (Supervised)
- Është një grumbull i pemëve vendimmarrëse.

- I përshtatshëm për të dhëna si numerike ashtu edhe kategorike.

- Kap marrëdhënie jo-lineare ndërmjet variablave dhe ndërveprime komplekse si “Gjinia” me “Niveli Akademik”.

- I qëndrueshëm ndaj noises dhe overfiting të modelit.

Avantazhe ndaj Regresionit Logjistik dhe SVM:

- Nuk ka nevojë për transformim të dhënash në formë numerike si regresioni logjistik.

- Është më i thjeshtë për tu trajnuar dhe më i përshtatshëm për dataset-e të përmasave mesatare krahasuar me SVM.

### 2. XGBoost (Supervised)
- Algoritëm i fuqishëm që ndërton pemë vendimmarrëse në mënyrë graduale.

- Përfshin mekanizma të rregullimit (regularization) për të shmangur mbingarkesën.

- I trajton më mirë të dhënat me mungesa, ato kategorike dhe rastet me pabarazi klasash.

- Jep rëndësinë e karakteristikave (feature importance), duke ndihmuar në interpretimin e modelit.

Avantazhe ndaj Rrjeteve Neurale:

- Nuk kërkon shumë të dhëna.

- Trajnohet më shpejt dhe është më i kuptueshëm.

- I optimizuar për të dhëna tabelare si ato në dataset-in tonë.

### 3. ARIMA (Model Kohor)
- I përshtatshëm për parashikimin e “Numrit të Nxënësve” në seri kohore.

- Mund të aplikohet veçmas për çdo komunë, nivel akademik etj.

- Kap trendet dhe sezonalitetin në të dhënat vjetore.

Avantazhe ndaj LSTM dhe Prophet:

- Më i thjeshtë dhe më i kuptueshëm për dataset-e të vogla.

- Jep më shumë kontroll dhe interpretueshmëri statistikore.

### 4. K-Means Clustering (Unsupervised)
- I përshtatshëm për segmentimin e komunave me profile të ngjashme arsimore.

- Kërkon kodim të variablave kategorikë për të funksionuar siç duhet.

- Ndihmon në krijimin e politikave arsimore të bazuara në të dhëna.

Avantazhe ndaj Klasifikimit Hierarkik:

- Më i shpejtë për dataset-e të mëdha.

- Më i lehtë për t’u interpretuar dhe vizualizuar.

### 5. DBSCAN + t-SNE (Unsupervised + Vizualizim)
- DBSCAN krijon klastra me forma jo të rregullta dhe zbulon të dhëna jonormale.

- Nuk ka nevojë të caktohet numri i klasave si tek K-Means.

- t-SNE zvogëlon dimensionet për vizualizim në 2D/3D duke shfaqur më qartë strukturat e fshehura të të dhënave.

Avantazhe ndaj PCA + K-Means:

- t-SNE është metodë jo-lineare që shfaq më mirë ndarjet natyrore të të dhënave.

- DBSCAN është më efektiv në identifikimin e klasave të pazakonta ose “outliers”.

## Strategjitë e Përdorura për Përmirësim të Modeleve
- Në algoritmet si Random Forest dhe XGBoost është përdorur data augmentation për të rritur përformancën në ndarjen test/train.

- Të gjitha modelet janë testuar dhe vlerësuar me metrika të ndryshme:

Metrikat për vlerësimin e modeleve regresive:
- R² Score (vlerësim i përputhshmërisë së modelit)

- Mean Absolute Error (MAE)

- Mean Squared Error (MSE)

- Root Mean Squared Error (RMSE)

- Cross-Validated R² Score

## Krahasimi i Modeleve dhe Performanca
Pas testimeve dhe krahasimeve të rezultateve:

XGBoost i optimizuar (tuned) ka performuar më mirë se Random Forest dhe versioni bazik i XGBoost.
```
| **Metric** | **Untuned XGBoost** | **Tuned XGBoost** | **Random Forest** | **Best**          |
|------------|---------------------|-------------------|-------------------|-------------------|
| R²         | 0.96                | 0.97              | 0.97              | Tie               |
| MAE        | 270.56              | 170.81            | 128.56            | Random Forest     |
| MSE        | 392,776.03          | 273,361.44        | 300,765.91        | Tuned XGBoost     |
| RMSE       | 626.72              | 522.84            | 548.42            | Tuned XGBoost     |
```
ARIMA ka treguar performancë shumë të mirë vizuale dhe përputhje të lartë ndërmjet të dhënave të trajnimit dhe testimit.
Rezultatet e ARIMA (2022–2023):
```
MAE: 1618.84

RMSE: 1644.46

R² Score: 0.965
```
Modelet K-Means dhe DBSCAN tregojnë ndarje të qartë të të dhënave në klasterë të veçantë.

Për çdo klaster është analizuar ndarja sipas gjinisë, komunës, moshës dhe nivelit akademik.