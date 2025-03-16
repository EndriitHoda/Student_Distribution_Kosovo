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

## Përshkrimi i Dataseteve

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