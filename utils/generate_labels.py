import pandas as pd
from pathlib import Path
import openpyxl

def generate_labels_from_excel(input_excel_path, output_csv_path):
    print(f"📂 正在读取 Excel 文件: {input_excel_path}...")
    
    # 1. 直接读取 .xlsx 文件 (无比省心，不需要管 encoding)
    df = pd.read_excel(input_excel_path)

    # 2. 清理表头前后的多余空格，防止提取列时报错
    df.columns = [str(col).strip() for col in df.columns]

    # 3. ✨ 核心魔法：向下填充合并单元格导致的缺失序号 ✨
    df['序号'] = df['序号'].ffill()
    df['序号'] = df['序号'].astype(int)

    # 4. 清理所有的字符串，去除前后误敲的空格和换行符
    df['词目'] = df['词目'].astype(str).str.strip()
    df['出现位置'] = df['出现位置'].astype(str).str.strip()

    # 5. 只提取我们需要的 3 列（抛弃表格里其他可能的备注列）
    final_df = df[['序号', '词目', '出现位置']]

    # 6. 保存为深度学习管线要求的终极标准格式：无表头(header=False)，GBK编码
    final_df.to_csv(output_csv_path, encoding='gbk', index=False, header=False)
    
    print(f"\n🎉 转换大功告成！标准文件已生成: {output_csv_path}")
    print("\n🧐 预览前 5 行完美对齐的数据：")
    print(final_df.head(5).to_string(index=False))

if __name__ == '__main__':
    # 填入你真实的 Excel 文件名
    input_file = r"C:\Users\capg303\Desktop\Project\手形\左右手标记.xlsx" 
    
    # 我们的 DataLoader 唯一认准的终极文件名
    output_file = "labels.csv"
    
    generate_labels_from_excel(input_file, output_file)