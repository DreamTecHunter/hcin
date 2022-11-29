clear;
wave_file = dir("wav_rec\*.wav");
for idx = 1:length(wave_file)
    fname = wave_file(idx).name;
    str = strfind(fname, '_');
    speaker_id = fname(1:str(1)-1);
    wave_file(idx).speaker_id = speaker_id;
    word_id = fname(str(1)+1:str(2)-1); 
    wave_file(idx).word_id = word_id;
    word_rep = fname(str(2)+1:end-4);
    wave_file(idx).word_rep = word_rep;
end
save("main.mat", "wave_file",'-mat')
