function [full_patient_list, isCOVID] = leaveOneOutSetup(patientNumber)
%  LEAVEONEOUTSETUP Function for creating file structure and contents for leave 'patientNumber' 
% out
% 
% validation
    % If the directory already exists, remove it
    if(isfolder('leave_out'))
        rmdir ('leave_out', 's');
    end
    
    % Create the directory
    if(~isfolder("leave_out"))
         mkdir leave_out/validation/covid
         mkdir leave_out/validation/noncovid
         mkdir leave_out/train/covid
         mkdir leave_out/train/noncovid
    end
    
    % COVID Spreadsheet
    COVID_sheet = readtable('COVID-CT-MetaInfo.xlsx','ReadVariableNames',false);
    % Get number of COVID images.
    covid_size = size(COVID_sheet,1);
    
    % NonCOVID Spreadsheet
    nonCOVID_sheet = readtable('NonCOVID-CT-MetaInfo.csv','ReadVariableNames',false);
    % Get number of nonCOVID images.
    noncovid_size = size(nonCOVID_sheet,1);
    
    % COVID patient IDs
    covid_patient_list = unique(COVID_sheet.Var2);
    % Get rid of blank first cell.;
    covid_patient_list = covid_patient_list(2:end);
    
    % NonCOVID patient IDs
    noncovid_patient_list = unique(nonCOVID_sheet.Var3);
    
    % Create the full patient list by concatenating the two lists.
    full_patient_list = [covid_patient_list ; noncovid_patient_list];
    
    % Total number of patients.
    num_patients = size(full_patient_list,1);
    
    % Path for covid validation.
    val_covid_path = fullfile('leave_out', 'validation','covid');
    % Path for noncovid validation.
    val_noncovid_path = fullfile('leave_out', 'validation','noncovid');
    % Path for covid training.
    train_covid_path = fullfile('leave_out', 'train','covid');
    % Path for noncovid training.
    train_noncovid_path = fullfile('leave_out', 'train','noncovid');
    
    % Path for covid images.
    covid_images = fullfile('images_and_split','CT_COVID');
    % Path for noncovid images.
    noncovid_images = fullfile('images_and_split','CT_NonCOVID');
    
    % Move every image into the training for now.
    
    % COVID images.
    copyfile(fullfile(covid_images), train_covid_path);
    % nonCOVID images.
    copyfile(fullfile(noncovid_images, '*'), train_noncovid_path);
    copyfile(fullfile(noncovid_images, '*'), train_noncovid_path);
%% 
% Code for filling folders with correct patient data
    if (patientNumber ~= 0)
        k = patientNumber;
        patient_id = full_patient_list{k};

        % Bool to check which spreadsheet we should look at.
        % Check for covid or noncovid patient.
        if(ismember(patient_id, COVID_sheet.Var2))
            isCOVID = 1;
        else
            isCOVID = 0;
        end

        % Find all images associated with the patient.
        images_to_move = strings(0);

        % Go through the COVID images and get the images that need to be moved.
        if(isCOVID == 1)
            for i = 1:covid_size
                if(strcmp(COVID_sheet.Var2{i},patient_id) == 1)
                    images_to_move = [images_to_move ; COVID_sheet.Var1{i}];
                end
            end
        end

        % Go through the nonCOVID images and get the images that need to be moved.
        if(isCOVID == 0)
            for i = 1:noncovid_size
                if(strcmp(nonCOVID_sheet.Var3{i},patient_id) == 1)
                    images_to_move = [images_to_move ; nonCOVID_sheet.Var2{i}];
                end
            end
        end

        % Get number of images to move.
        move_num = size(images_to_move,1);

        % Move the COVID patient's images into the validation folder.
        if(isCOVID == 1)
            for i = 1:move_num
                file_name = images_to_move(i);
                movefile(fullfile(train_covid_path, file_name + "*"), val_covid_path);
            end
            % Move dummy image to validation Covid
        end

        % Move the nonCOVID patient's images into the validation folder.
        if(isCOVID == 0)
            for i = 1:move_num
                file_name = images_to_move(i);
                movefile(fullfile(train_noncovid_path, file_name + "*"), val_noncovid_path);
            end
            % Move dummy image to validation nonCovid
        end
    else
        isCOVID = 1;
    end
end %End Function