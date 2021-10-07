library(tidyverse)
library(skimr)
df <- read.csv("C:\\Users\\Arjun\\Downloads\\Datasets\\Forest Cover\\Forest Cover.csv")
glimpse(df)
nrow(df) - nrow(distinct(df))
any(is.na(df))
skim(df)
set.seed(1808)

dff <- (df %>%
         group_by(Cover_Type) %>%
         sample_n(size = 1000) %>%
         ungroup() %>%
         gather(Wilderness_Area, Wilderness_Value, 
                Wilderness_Area1:Wilderness_Area4) %>% 
         filter(Wilderness_Value >= 1) %>%
         select(-Wilderness_Value) %>%
         mutate(Wilderness_Area = str_extract(Wilderness_Area, '\\d+'),
                Wilderness_Area = str_replace_all(Wilderness_Area,
                                                  c(`1` = 'Rawah',
                                                    `2` = 'Neota',
                                                    `3` = 'Comanche Peak',
                                                    `4` = 'Cache la Poudre')),
                Wilderness_Area = as.factor(Wilderness_Area)) %>%
         gather(Soil_Type, Soil_Value, Soil_Type1:Soil_Type40) %>%
         filter(Soil_Value == 1) %>%
         select(-Soil_Value) %>% 
         mutate(Soil_Type = as.factor(str_extract(Soil_Type, '\\d+'))) %>%
         mutate(Cover_Type = str_replace_all(Cover_Type,
                                             c(`1` = 'Spruce/Fir',
                                               `2` = 'Lodgepole Pine',
                                               `3` = 'Ponderosa Pine',
                                               `4` = 'Cottonwood/Willow',
                                               `5` = 'Aspen',
                                               `6` = 'Douglas Fir',
                                               `7` = 'Krummholz')),
                Cover_Type = as.factor(Cover_Type)) %>%
         select(Cover_Type:Soil_Type, Elevation:Slope,
                Hillshade_9am:Hillshade_3pm, Vertical_Distance_To_Hydrology,
                Horizontal_Distance_To_Hydrology:Horizontal_Distance_To_Fire_Points))
glimpse(dff)
skim(dff)
