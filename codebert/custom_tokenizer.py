custom_special_tokens_dict = {'additional_special_tokens': ['<mutant_op>', '</mutant_op>',
                            '<original_code>','</original_code>', '<mutant_code>', '</mutant_code>',
                            '<subsuming>' , '</subsuming>']}
mutant_operators =[
    "IdentifierMutator-Variable",
    "IdentifierMutator-Literal",
    "ArrayAccessMutator",
    "UnaryOperatorMutator",
    "BinaryOperatorMutator",
    "TypeReferenceMutator",
    "MethodCallMutator",
    "IdentifierMutator-ThisAccess",
    "IdentifierMutator-Conditional",
    "AssignmentMutator",
    "FieldReferenceMutator",
    "BrokenCSV"
]

import json                    
abstraction_vocab = json.load(open("../data/vocab_abstraction.json"))