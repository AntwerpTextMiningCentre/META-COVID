# META-COVID

```
from transformers import RobertaTokenizer
from demo_workflow import BertForMultiLabelClassification, DemoFunctions
```

```
# Step 1: Loading the model
MODEL_PATH = 'PieterFivez/metacovid-parliament'
model = BertForMultiLabelClassification.from_pretrained(MODEL_PATH)
model.eval()
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
```

```
# Step 2: Demo
Demo = DemoFunctions(model, tokenizer)

text = "Mijnheer de minister , op 29 april 2020 , een halve maand geleden dus , verklaarde u tijdens een interview met de RTBF dat de regering aan Defensie de opdracht had gegeven achttien miljoen mondmaskers te bestellen . Vorige week mocht ik in de commissie Legeraankopen van de Staf vernemen op welke manier Defensie die bestelling had aangepakt . Mijnheer de minister , ik moet eerlijk toegeven dat ik er vrij veel vertrouwen in had . Ik had er vertrouwen in dat voor een keer in ons land eens iets efficiënt zou worden aangepakt . Dat was echter buiten de waard of buiten de Belgische Staat gerekend . Blijkbaar is er immers weinig of geen achtergrondcontrole van de potentiële buitenlandse leveranciers gebeurd , waardoor minstens moet worden opgemerkt dat die leveranciers balanstechnisch van bedenkelijke aard zijn ."
probabilities = Demo.predict(text)
predicted_labels = Demo.decode_predictions_parliament(probabilities)
```
