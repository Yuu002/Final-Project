setwd("C:/project")

# library ทั้งหมด
library(readxl)
library(Hmisc)
library(lmtest)
library(car)
library(MASS)
library(olsrr)

# นำ data เข้า
data <- read_excel("AGB_Data.xlsx")
str(data) # เช็ค type
colSums(is.na(data)) # เช็ค missing values
sum(duplicated(data)) # เช็คค่าซ้ำ

# ตรวจค่าผิดปกติ Outlier
boxplot(data$AGB, main = "Boxplot of AGB", col = "skyblue")
boxplot(data[, c("NDVI","TNDVI","SR","SAVI","MSAVI2")], main = "Boxplot of Vegetation indices", col = "skyblue")

# แผนภาพการกระจายของตัวแปรแต่ละคู่
pairs(AGB~NDVI+TNDVI+SR+SAVI+MSAVI2, data=data)
--------------------------------------------------------------------------------------------------------------------------

# ค่าสัมประสิทธิ์สหสัมพันธ์กับ AGB
cor.test(data$NDVI, data$AGB)
cor.test(data$TNDVI, data$AGB)
cor.test(data$SR, data$AGB)
cor.test(data$SAVI, data$AGB)
cor.test(data$MSAVI2, data$AGB)

# คำนวน matrix ค่าสัมประสิทธิ์สหสัมพันธ์
cor(data)
rcorr(as.matrix(data))

# สร้างสมการ Linear regression วิธี stepwise regression เพื่อเลืกตัวแปรที่เหมาะสม
model <- lm(AGB~NDVI+TNDVI+SR+SAVI+MSAVI2, data=data)
summary(model)

STPfit.p <- ols_step_both_p(model)
STPfit.p

# ทดสอบสมมติฐานเกี่ยวกับสัมประสิทธิ์ถดถอยแต่ละตัว
model_1 <- lm(AGB~TNDVI+MSAVI2, data=data)
summary(model_1)

model_2 <- lm(AGB~TNDVI, data=data)
summary(model_2)

# สมการสุดท้าย
fit.model <- lm(AGB~TNDVI, data=data)
summary(fit.model)

# ตรวจสอบข้อตกลงเบื้องต้นของสมการ
shapiro.test(data$AGB) # แจกแจงไม่ปกติ

# เริ่มทำให้โดยแปลงค่า Y
# แปลงค่า AGB(sqrtAGB)
sqrtAGB <- sqrt(data$AGB)

# ตรวจค่าผิดปกติ Outlier
boxplot(sqrtAGB, main = "Boxplot of AGB", col = "skyblue")

# แผนภาพการกระจายของตัวแปรแต่ละคู่
pairs(sqrtAGB~NDVI+TNDVI+SR+SAVI+MSAVI2, data=data)

# สร้างสมการ Linear regression วิธี stepwise regression เพื่อเลืกตัวแปรที่เหมาะสม
model_c <- lm(sqrtAGB~NDVI+TNDVI+SR+SAVI+MSAVI2, data=data)
summary(model_c)

STPfit.p <- ols_step_both_p(model_c)
STPfit.p

# ทดสอบสมมติฐานเกี่ยวกับสัมประสิทธิ์ถดถอยแต่ละตัว
model <- lm(sqrtAGB~TNDVI, data=data)
summary(model)

# สมการสุดท้าย
fit.model <- lm(sqrtAGB~TNDVI, data=data)
summary(fit.model)

# ตรวจสอบข้อตกลงเบื้องต้นของสมการ
shapiro.test(sqrtAGB) # แจกแจงปกติ

# ตรวจสอบความคลาดเคลื่อนมีการแจกแจงปกติหรือไม่
shapiro.test(fit.model$residual) # แจกแจงปกติ

# ตรวจสอบความคลาดเคลื่อนมีค่าเฉลี่ยเท่ากับ 0 หรือไม่
t.test(fit.model$residual, mu=0) # ความคลาดเคลื่อนมีค่าเฉลี่ยเท่ากับ 0

# ตรวจสอบความคลาดเคลื่อนมีความแปรปรวนคงที่หรือไม่
bptest(fit.model) # ความแปรปรวนคงที่

# ความคลาดเคลื่อนมีความสัมพันธ์ในตัวเองหรือไม่
durbinWatsonTest(fit.model) # มีความสัมพันธ์ในตัวเอง

# ลองทำนาย
range(data$AGB)
pred <- predict(fit.model)^2
actual <- data$AGB
head(data.frame(Actual = actual, Predicted = round(pred, 2)), 10)

rmse <- sqrt(mean((actual - pred)^2))
mae <- mean(abs(actual - pred))
cat("RMSE = ", round(rmse, 2), "\nMAE =", round(mae, 2))
summary(fit.model)

