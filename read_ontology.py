from rdflib.namespace import Namespace
from rdflib import Graph, RDF, RDFS, OWL


class Concept:
    def __init__(self, concept_name, metadata):
        self.name = concept_name
        self.metadata = metadata

class Topic:
    def __init__(self, name):
        self.name = name
        self.concepts = []

    def add_concept(self, concept):
        self.concepts.append(concept)

class MetaCovidOntology:
    def __init__(self, owl_file_path):
        
        self.owl_file_path = owl_file_path
        
        # Define namespaces within the class
        self.BASE = Namespace("http://example.org/metacovid/")
        self.UMLS = Namespace("http://example.org/metacovid/UMLS/")
        self.CAP = Namespace("http://example.org/metacovid/CAP/")
        self.EUROVOC = Namespace("http://example.org/metacovid/Eurovoc/")
        self.METADATA = Namespace("http://example.org/metacovid/metadata/")
        
        self.metadata = {}
        self.topics = []
        self._load_data()

    def _load_data(self):
        
        g = Graph()
        g.parse(self.owl_file_path)

        # Extract metadata
        for ontology in ["UMLS", "CAP", "EuroVoc"]:
            ontology_namespace = None
            if ontology == "UMLS":
                ontology_namespace = self.UMLS["Metadata"]
            elif ontology == "CAP":
                ontology_namespace = self.CAP["Metadata"]
            elif ontology == "EuroVoc":
                ontology_namespace = self.EUROVOC["Metadata"]
            
            self.metadata[ontology] = {}
            for pred, obj in g.predicate_objects(subject=ontology_namespace):
                pred_name = str(pred).split("/")[-1]
                if pred_name != "22-rdf-syntax-ns#type":
                    self.metadata[ontology][pred_name] = str(obj)

        # Extract topics and associated concepts
        for subject in g.subjects(RDF.type, OWL.Class):
            topic_name = str(subject).split("/")[-1].replace("_", " ")
            topic_obj = Topic(topic_name)
            
            for individual in g.subjects(RDF.type, subject):
                concept_name = None
                concept_metadata = {}
                for pred, obj in g.predicate_objects(subject=individual):
                    pred_name = str(pred).split("/")[-1]
                    # Exclude the '22-rdf-syntax-ns#type' key for concepts as well
                    if pred_name != "22-rdf-syntax-ns#type":
                        if pred == RDFS.label:
                            concept_name = str(obj)
                        else:
                            concept_metadata[pred_name] = str(obj)
                                    # Set the ontology to which the concept belongs in its metadata
                if "UMLS" in str(individual):
                    concept_metadata["ontology"] = "UMLS"
                elif "CAP" in str(individual):
                    concept_metadata["ontology"] = "CAP"
                elif "EuroVoc" in str(individual):
                    concept_metadata["ontology"] = "EuroVoc"
                
                topic_obj.add_concept(Concept(concept_name, concept_metadata))
                
            # Append the topic to the topics list
            self.topics.append(topic_obj)

        # sort topics by MC index
        resort_indices = sorted(range(len(self.topics)), key=lambda x: int(self.topics[x].name.split(':')[0][2:])) 
        self.topics = [self.topics[i] for i in resort_indices]

    def get_metadata(self):
        return self.metadata

    def get_topics(self):
        return self.topics
