import argparse
import pandas as pd
import torch
from molscribe import MolScribe
from rdkit import Chem
from DECIMER import predict_SMILES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='../data/test', type=str, help='img dir')
    parser.add_argument('--indices_path', default='../data/indices.csv', type=str, help='indices path')
    parser.add_argument('--output_path', default='../predictions.csv', type=str, help='output path')
    args = parser.parse_args()

    df = pd.read_csv(args.indices_path)

    id = df['id'].tolist()
    
    MORE_IMAGES = []

    for i in id:
      MORE_IMAGES.append(args.img_dir + f'/{i}.png')
    
    MODEL_PATH = '/opt/weis/swin_base_char_aux_1m680k.pth'
    model = MolScribe(MODEL_PATH, device)
    
    MODEL_PATH_2 = '/opt/weis/swin_base_char_aux_1m.pth'
    model2 = MolScribe(MODEL_PATH_2, device)
    
    MODEL_PATH_3 = '/opt/weis/swin_base_char_aux_200k.pth'
    model3 = MolScribe(MODEL_PATH_3, device)
    
    predictions = model.predict_image_files(MORE_IMAGES, return_atoms_bonds=False, return_confidence=True)
    smi2 = 'N=C1N(C2CCCCC2)CCN1[S@SP3](=O)(=O)c1ccc(CCNC(=O)c2ccccn2)cc1'
    
    lst = []
    fix_list = []
    
    for i in range(len(predictions)):
        smi1 = predictions[i]['smiles']
        confidence = predictions[i]['confidence']
        mol = Chem.MolFromSmiles(smi1)
            
        if mol != None and confidence > 0.3:
          lst.append(smi1)
        else:
          lst.append('')
          fix_list.append(id[i])
    
    MORE_IMAGES = []
    for i in fix_list:
      MORE_IMAGES.append(args.img_dir + f'/{i}.png')
    
    if MORE_IMAGES != []:
        predictions = model2.predict_image_files(MORE_IMAGES, return_atoms_bonds=False, return_confidence=True)
        fix_list2 = []
        
        for i in range(len(predictions)):
            smi1 = predictions[i]['smiles']
            confidence = predictions[i]['confidence']
            mol = Chem.MolFromSmiles(smi1)
                
            if mol != None and confidence > 0.3:
              lst[id.index(fix_list[i])] = smi1
            else:
              fix_list2.append(fix_list[i])
        
        MORE_IMAGES = []
        for i in fix_list2:
          MORE_IMAGES.append(args.img_dir + f'/{i}.png')
        
        if MORE_IMAGES != []: 
            predictions = model3.predict_image_files(MORE_IMAGES, return_atoms_bonds=False, return_confidence=True)
            
            for i in range(len(predictions)):
                smi1 = predictions[i]['smiles']
                confidence = predictions[i]['confidence']
                mol = Chem.MolFromSmiles(smi1)
                    
                if mol != None and confidence > 0.3:
                    lst[id.index(fix_list2[i])] = smi1
            
    k = 0
    k_max = 10**10
    for i in range(len(lst)):
        if lst[i] == '' and k < k_max:
            k += 1
            smi1 = predict_SMILES(args.img_dir + f'/{id[i]}.png')
            mol = Chem.MolFromSmiles(smi1)
            
            if mol != None:
                lst[i] = smi1
            else:
                lst[i] = smi2
        elif lst[i] == '':
            lst[i] = smi2
    
    df = pd.DataFrame.from_dict({"id": id, "smiles": lst})
    df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
