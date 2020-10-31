
# Read RDS file and export it into csv

file1 <- "../data/IMPROVE_2011-2013_filterSubtractionV2.RDS"
file2 <- "../data/matched_std_2784_baselined_filter_subtracted.RDS"

con1 <- gzfile(file1)
spec1 <- readRDS(con1)
  
con2 <- gzfile(file2)
spec2 <- readRDS(con2)
  
write.csv(spec1, "../data/IMPROVE_2011-2013_filterSubtractionV2.csv")
write.csv(spec2, "../data/matched_std_2784_baselined_filter_subtracted.csv")


