def DfCustomPrintFormat(df):
    return "\n      ".join(df.head().to_string().split("\n"))
