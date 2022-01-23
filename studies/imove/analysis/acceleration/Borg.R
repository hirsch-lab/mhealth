# Borg 
#### SETTINGS ###########################################
# rm(list = ls())

#### LIBRARIES ###########################################
library(tidyverse)

#### PATHS ###########################################
data_path = '/Users/julien/GD/ACLS/TM/DATA/Borg/'
plots_path = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 MODULE/TM/OUTPUT/plots/'

#### SOURCE SCRIPTS ###########################################
#### LOAD DATA  ###########################################
borg_file = paste(data_path, 'iMove_Borg_JB.csv', sep='')
df = read.delim(borg_file, sep=",", dec=".", header=TRUE) # row.names = 1
head(df)

#### PROCESS  ###########################################
# `Borg_D1_pre`, `Borg_D2_pre`, `Borg_D3_pre`	
# `Borg_D1_post`,	`Borg_D2_post`,	`Borg_D3_post`

## prep
prep = df %>%
            as_tibble() %>%
            mutate_if(is.character, as.factor) %>%
  
    # calculate exertion: post - pre (increased exertion is pos value)
            mutate(exertion_D1 = Borg_D1_post - Borg_D1_pre) %>% # NAs -> NAs
            mutate(exertion_D2 = Borg_D2_post - Borg_D2_pre) %>%
            mutate(exertion_D3 = Borg_D3_post - Borg_D3_pre)
  
# borg
# gather 6 to 1
borg = prep %>%             
              select(-(exertion_D1:exertion_D3)) %>% # delete unneeded cols
              gather(`Borg_D1_pre`, `Borg_D2_pre`, `Borg_D3_pre`,
              `Borg_D1_post`, `Borg_D2_post`, `Borg_D3_post`,
              key = KEY, value = "borg") %>%

            # separate KEY (Borg_D1_pre) at _ to create day-variable and pre_post-variable
            separate(KEY, c("trash", "day", "pre_post"), '_') %>%
            select(-(trash)) %>%
            rename(measurement=pre_post) %>%
            mutate_if(is.character, as.factor) # measurement: chr --> fct 
head(borg) 

# exertion (all 3 days)
exertion = prep %>%
          select(-(Borg_D1_pre:Borg_D3_post)) %>% # delete unneeded cols
          gather(`exertion_D1`, `exertion_D2`, `exertion_D3`,
          key = day, value = "exertion") %>%
  
          # separate KEY-var: day (exertion_D1) at _ to create trash and day-variable
          separate(day, c("trash", "day"), '_') %>%
          select(-(trash)) %>%
          mutate_if(is.character, as.factor) # day: chr --> fct 
head(exertion)

# exertion_D23 (only D2, D3)
exertion_D23 = exertion %>%
  filter(day!="D3")


#### DELETE  ###########################################
rm(df, prep)

#### ANALYSIS: BORG  ###########################################
sapply(borg, class)
sapply(borg, function(x) sum(is.na(x)))

# Statistics
borg %>%
  drop_na(borg) %>%
  filter(day != "D1") %>% # remove D1 obs
  group_by(day, measurement) %>%
  summarise_at(vars(borg), list(mittelwert = mean, varianz = var))

#### ANALYSIS: EXERTION  ###########################################

# for day 2 or day 3
# patient_IDs of patients who perceived increased exertion of 2 and more
exertion_D23$patient_ID[exertion_D23$exertion >= 2] %>% na.exclude() %>% sort()
# patient_IDs of patients who perceived increased exertion of 4 and more
exertion_D23$patient_ID[exertion_D23$exertion >= 4] %>% na.exclude() %>% sort()

# Correlation
# cor(x=exertion_D23$BMI, y=exertion_D23$exertion, method = c("pearson"))
cor.test(x=exertion_D23$BMI, y=exertion_D23$exertion, method=c("pearson"))

#### GRAPHICS: BORG ###########################################
# graphics settings
# geom_boxplot(outlier.colour="black", outlier.shape=16,
#              outlier.size=2, notch=FALSE)

# axis names 
y_axis_name = "Borg scale"
xaxislabel = "Time of measurement"
"Self-examination of perceived exertion"

# 1) fig___BORG_measurement_vs_borg.png 
# measurement vs borg
png(paste0(plots_path,"fig___BORG_measurement_vs_borg.png")) # save as png to specified folder
p <- ggplot(borg, aes(x=measurement, y=borg)) + 
  geom_boxplot() + 
  geom_point(position = position_jitter(w = 0.3, h = 0.05)) +
  scale_y_continuous(breaks = c(0, 1, 2, 3, 4, 5, 6,7,8,9,10), # muss 0 bis 10 angeben
                     labels=c("0 Rest", 	"1 very, very easy", 	"2 easy", 	"3 moderate", 	
                              "4 somewhat hard", 	"5 hard", 	"6 -", 	"7 very hard", 	
                              "8 -", 	"9 -", 	"10 maximal"),
                     name=y_axis_name
                     ) +
  scale_x_discrete(name=xaxislabel,
                   labels=c("after", "before"))
p # printet nicht in console
print(fig___BORG_measurement_vs_borg.png) # file name for png
dev.off()

# 2) fig___BORG_day_vs_borg.png 
# day vs borg. color=pre_post
png(paste0(plots_path,"fig___BORG_day_vs_borg.png")) # save as png to specified folder
p <- ggplot(borg, aes(x=day, y=borg, color=measurement)) + 
  geom_boxplot() + 
  geom_point(position = position_jitter(w = 0.3, h = 0.05)) +
  scale_y_continuous(breaks = c(0, 1, 2, 3, 4, 5, 6,7,8,9,10), # muss 0 bis 10 angeben
                     labels=c("0 Rest", 	"1 very, very easy", 	"2 easy", 	"3 moderate", 	
                              "4 somewhat hard", 	"5 hard", 	"6 -", 	"7 very hard", 	
                              "8 -", 	"9 -", 	"10 maximal"),
                     name=y_axis_name
                     )
p
print(fig___BORG_day_vs_borg.png) # file name for png
dev.off()

#### GRAPHICS: EXERTION  (only for D2, D3) ###########################################
# axis names 
xaxis_exertion = "Exertion after DEMMI-exercises"
yaxis_hist = "# Patients"

# for vertical mean line: must exclude all NAs!
exertion_D23_noNA = exertion_D23$exertion[-which(is.na(exertion_D23$exertion))] # exclude all NAs
mean(exertion_D23_noNA)


# 1) Basic histogram: fig___BORG_exertionD23_histo.png 
png(paste0(plots_path,"fig___BORG_exertionD23_histo.png")) # save as png to specified folder

ggplot(exertion_D23, aes(x=exertion)) + 
  geom_histogram(binwidth=1, 
                 color="black",
                 fill="white") +
  # add vertical mean line
  geom_vline(aes(xintercept=mean(exertion_noNA)),
             color="blue", linetype="dashed", size=1) +
  # add density plot
  geom_density(alpha=.2, fill="#FF6666") + # warum so tief?
  scale_y_continuous(#breaks = c(),
    #labels=c(),
    name=yaxis_hist
  ) +
  scale_x_continuous(name=xaxis_exertion
  ) +
  
  # annotate
  annotate(geom="text", x=-2.8, y=45, label="vitality gain",
           color="red") +
  annotate(geom="text", x=4, y=45, label="increased exertion",
           color="red")

print(fig___BORG_exertionD23_histo.png) # file name for png
dev.off()


# Scatterplot: fig___BORG_exertionD23_scatter_BMI_exertionD23.png 
png(paste0(plots_path,"fig___BORG_exertionD23_scatter_BMI_exertionD23.png")) 

ggplot(exertion_D23, aes(x=BMI, y=exertion)) + 
  geom_point(size= 0.5) +
  geom_smooth(method="lm", se=TRUE, fullrange=FALSE, level=0.95) +
  xlim(15, 40) +
  geom_vline(xintercept = 24   , linetype="dotted", color="green") + # good
  geom_vline(xintercept = 30.9 , linetype="dotted", color="green")   # booc
# cf: https://thegeriatricdietitian.com/bmi-in-the-elderly/

print(fig___BORG_exertionD23_scatter_BMI_exertionD23.png) # file name for png
dev.off()


#### a ###########################################










