### Prep_KMA.py generates KMA command to execute the KMA mapping. 

def prep_kma(path_forward_reads, path_reverse_reads, sample_name, kma_script_model, kma_path, db_path, out_path):

    #Extract data from input file and create script: 
    kma_script = open(kma_script_model, 'r')
    #out_file = open(sample_name+'_kma_Mapping.sh', 'w')
    
    #Read kma file and modify the kma command: 
    for line in kma_script: 
        tmp_line = line.replace('path_R1', path_forward_reads)
        tmp_line2 = tmp_line.replace('path_R2', path_reverse_reads)
        tmp_line3 = tmp_line2.replace('sample_name', sample_name)
        tmp_line4 = tmp_line3.replace('kma_dir', kma_path)
        tmp_line5 = tmp_line4.replace('out_dir', out_path)
        final = tmp_line5.replace('database', db_path)
    
    #Close output file: 
    #out_file.close()
    kma_script.close()

    return final
