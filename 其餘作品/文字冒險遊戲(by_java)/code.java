import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class code {
    static int P_HP,P_ATK,P_CT,P_LV,B_LV,B_HP,B_ATK,B_CT;
    static String B_Name,P_Name;
    private static Random random = new Random();

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        userLogin(sc);
    }
    //登入函式
    private static void userLogin(Scanner sc) {
        System.out.print("輸入使用者名稱(user name): ");
        String user = sc.nextLine();
        boolean isUser = isUserDuplicate(user);
        if (!isUser) {
            System.out.print("歡迎新玩家: "+user+"\n");
            appendUserToFile(user);
            createCharacter(user);
            chooseOrCreateCharacter(user, isUser);
        }
        else{
            System.out.print("歡迎回來: "+user+"\n");
            chooseOrCreateCharacter(user, isUser);         
        }
    }
    //確認是否新使用者
    private static boolean isUserDuplicate(String user) {
        try (BufferedReader reader = new BufferedReader(new FileReader("user.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (user.equals(line)) {
                    return true;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }
    //新增使用者
    private static void appendUserToFile(String user) {
        try (FileWriter writer = new FileWriter("user.txt", true)) {
            writer.write(user + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //選角創角
    private static void chooseOrCreateCharacter(String user, boolean isUser) {
        Scanner sc=new Scanner(System.in);
        while (true) {
            System.out.print("\n[Z] 讀取角色 / [X] 創新角色 / [Q] 退出 :");
            String choice = sc.next();
            if (choice.equalsIgnoreCase("Z")) {
                readCharacterFromFile(user);
                System.out.print("\n選擇要使用的角色 (輸入編號): ");
                int n=sc.nextInt();
                chooseCharacter(user,n);
                randomBoss();
                fight();
            } else if (choice.equalsIgnoreCase("X")) {
                createCharacter(user); // 正確傳遞了 Scanner 對象和用戶名稱
                chooseOrCreateCharacter(user, isUser);
            } else if (choice.equalsIgnoreCase("Q")) {
                return;
            } else {
                System.out.println("請輸入有效的選項");
            }
        }
    }
    //RAND函式RAND(min,max)
    private static int RAND(int min, int max) {
            return random.nextInt((max - min) + 1) + min;
    }
    // 創建角色的函式
    private static void createCharacter(String user) {
        Scanner sc=new Scanner(System.in);
        System.out.print("\n選擇職業: (輸入編號)\n");
        System.out.print("[1].戰士\n[2].射手\n[3].刺客\n[4].牧師\n[5].潘朵拉\n");

        int professionChoice = sc.nextInt(); 
        String professionName = "";
        switch (professionChoice) {
            case 1:
                professionName = "戰士";
                sword(user);
                break;
            case 2:
                professionName = "射手";
                arrow(user);
                break;
            case 3:
                professionName = "刺客";
                knife(user);
                break;
            case 4:
                professionName = "牧師";
                stick(user);
                break;
            case 5:
                professionName = "潘朵拉";
                pandora(user);
                break;
            default:
                System.out.println("請輸入正確選項");
                createCharacter(user);
                return;
        }

        // 寫入選擇的職業到專屬的.txt檔案中
        writeToUserFile(user, professionName);
        System.out.print("已建立:"+professionName+"\n");
    }
    // 寫入選擇的職業到專屬的.txt檔案中
    private static void writeToUserFile(String user, String profession) {
        String fileName = user + ".txt";
        try (FileWriter writer = new FileWriter(fileName, true)) {
            //writer.write("職業：" + profession + "\n");
            writer.write("\n"); // 換行
            // 其他角色相關資訊的初始化
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    // 選擇Boss的函式(導入Boss數據的)
    private static void randomBoss() {
        int n=RAND(1,10); 
        String[] data = ReadBoss("Boss.txt",n);
        B_Name=data[0];
        B_LV=Integer.parseInt(data[1]);
        B_HP=Integer.parseInt(data[2]);
        B_ATK=Integer.parseInt(data[3]);
        B_CT=Integer.parseInt(data[4]);
        System.out.print("\n\nBoss:"+B_Name+" "+B_LV+" "+B_HP+" "+B_ATK+" "+B_CT+" \n");
    }
    // 選擇Boss的函式(找Boss數據的)
    private static String[] ReadBoss(String fileName, int lineNumber) {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            for (int i = 1; i < lineNumber; i++) {
                reader.readLine(); // 跳過前 lineNumber - 1 行
            }
            String line=reader.readLine();
            return line.split("\\s+");
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
    // 讀取角色的函式
    private static void readCharacterFromFile(String user){
        System.out.print("\n");
        String fileName = user + ".txt";
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;int n=1;
            while ((line = reader.readLine()) != null) {
                {System.out.print(n+". "+line+"\n");}
                n++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //導入角色數據(導入Player數據的)
    private static void chooseCharacter(String user,int n){
        String fileName=user+".txt";
        String[] data = ReadPlayer(fileName,n);
        P_Name=data[0];
        P_LV=Integer.parseInt(data[1]);
        P_HP=Integer.parseInt(data[2]);
        P_ATK=Integer.parseInt(data[3]);
        P_CT=Integer.parseInt(data[4]);
    }
    //選擇Player的函式(找Player數據的)
    private static String[] ReadPlayer(String fileName, int lineNumber){
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            for (int i = 1; i < lineNumber; i++) {
                reader.readLine(); // 跳過前 lineNumber - 1 行
            }
            String line = reader.readLine();
            return line.split("\\s+");
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }    
    }
    // 計算該帳號已經有多少角色
    private static int countCharacters(String user) {
        String fileName = user + ".txt";
        int count = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            while (reader.readLine() != null) {
                count++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return count;
    }
    //生成戰士
    private static void sword(String F) {
        F+=".txt";
        try (FileWriter writer = new FileWriter(F,true)) {
            int LV=RAND(10,80);
            int STR=RAND(32,56);
            int VIT=RAND(32,56);
            int LUC=RAND(32,56);
            int HP=LV*100+VIT*100+STR*50;
            int ATK=LV*10+STR*8+VIT;
            int CT=(LV+LUC)/2;
            writer.write("戰士 "+LV+" "+HP+" "+ATK+" "+CT+" "+STR+" "+VIT+" "+LUC+" ");
            P_Name="戰士";
            P_LV=LV;
            P_HP=HP;
            P_ATK=ATK;
            P_CT=CT;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //生成弓手
    private static void arrow(String F) {
        F+=".txt";
        try (FileWriter writer = new FileWriter(F,true)) {
            int LV=RAND(10,80);
            int STR=RAND(32,56);
            int VIT=RAND(8,32);
            int LUC=RAND(56,72);
            int HP=LV*100+VIT*100+STR*50;
            int ATK=LV*10+STR*8+VIT;
            int CT=(LV+LUC)/2;
            writer.write("射手 "+LV+" "+HP+" "+ATK+" "+CT+" "+STR+" "+VIT+" "+LUC+" ");
            P_Name="射手";
            P_LV=LV;
            P_HP=HP;
            P_ATK=ATK;
            P_CT=CT;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //生成刺客
    private static void knife(String F) {
        F+=".txt";
        try (FileWriter writer = new FileWriter(F,true)) {
            int LV=RAND(10,80);
            int STR=RAND(56,72);
            int VIT=RAND(8,32);
            int LUC=RAND(32,56);
            int HP=LV*100+VIT*100+STR*50;
            int ATK=LV*10+STR*8+VIT;
            int CT=(LV+LUC)/2;
            writer.write("刺客 "+LV+" "+HP+" "+ATK+" "+CT+" "+STR+" "+VIT+" "+LUC+" ");
            P_Name="刺客";
            P_LV=LV;
            P_HP=HP;
            P_ATK=ATK;
            P_CT=CT;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //生成牧師
    private static void stick(String F) { 
        F+=".txt";
        try (FileWriter writer = new FileWriter(F,true)) {
            int LV=RAND(10,80);
            int STR=RAND(32,56);
            int VIT=RAND(56,72);
            int LUC=RAND(8,32);
            int HP=LV*100+VIT*100+STR*50;
            int ATK=LV*10+STR*8+VIT;
            int CT=(LV+LUC)/2;
            writer.write("牧師 "+LV+" "+HP+" "+ATK+" "+CT+" "+STR+" "+VIT+" "+LUC+" ");
            P_Name="牧師";
            P_LV=LV;
            P_HP=HP;
            P_ATK=ATK;
            P_CT=CT;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //生成潘朵拉
    private static void pandora(String F) {
        F+=".txt";
        try (FileWriter writer = new FileWriter(F,true)) {
            int LV=1;
            int STR=1;
            int VIT=1;
            int LUC=50;
            int HP=LV*100+VIT*100+STR*50;
            int ATK=LV*10+STR*8+VIT;
            int CT=(LV+LUC)/2;
            writer.write("潘朵拉 "+LV+" "+HP+" "+ATK+" "+CT+" "+STR+" "+VIT+" "+LUC+" ");
            P_Name="潘朵拉";
            P_LV=LV;
            P_HP=HP;
            P_ATK=ATK;
            P_CT=CT;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //打鬥
    private static void fight(){
        System.out.print("玩家: "+P_Name+" "+P_LV+" "+P_HP+" "+P_ATK+" "+P_CT+" \n\n"); 
        int PlayerHP=P_HP,BossHP=B_HP;
        while(true){
            System.out.print("-----------------------------------\n\n");
            System.out.print(PlayerHP+"/"+P_HP+"   "+BossHP+"/"+B_HP+"\n");
            Scanner sc=new Scanner(System.in);
            System.out.print("[Z]打他 / [X]投降: ");
            String move=sc.next();
            if(move.equalsIgnoreCase("X")){
                int n=RAND(1,100);
                if(n>B_LV*0.6){
                    System.out.print("逃跑失敗\n");
                    continue;
                }
                else{
                    System.out.print("你成功逃走了\n");
                    break;
                }
            }
            int C,DAM;
            if(P_LV==1){BossHP=pandoramode(BossHP);}
            else{
                C=RAND(1,100);
                DAM=P_ATK;
                if(P_CT>C){DAM*=2;System.out.print("玩家: 爆擊!!造成"+DAM+"的傷害\n");}
                else{System.out.print("玩家: 打出"+DAM+"的傷害\n");}
                BossHP-=DAM;        
                
            }
            C=RAND(1,100);
            DAM=B_ATK;
            if(B_CT>C){DAM*=2;System.out.print("Boss: 爆擊!!造成"+DAM+"的傷害\n");}
            else{System.out.print("Boss: 打出"+DAM+"的傷害\n");}
            PlayerHP-=DAM;
            /*
            try {
                new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
            //*/
            if(PlayerHP<=0&&BossHP<=0){System.out.print("\n\n你們兩個都掛了...\n這是場不錯的決鬥...\n\n");break;}    
            else if(PlayerHP<=0){System.out.print("\n\n你已經死了\n\n");break;}
            else if(BossHP<=0){System.out.print("\n\nBoss已經死了\n\n");break;}  
        }
    }
    //潘朵拉的攻擊
    private static int pandoramode(int BossHP){
        int C=RAND(1,100);
        System.out.print("潘朵拉打開了魔盒\n");
        if(P_CT>C){
            System.out.print("僅存的【希望】從魔盒湧出，為世界帶來【希望】\n");
            B_ATK=0;
            return 0;
        }
        else{
            System.out.print("魔合裡的【惡夢】湧出，【惡夢】降臨\nBoss遭到反噬:損失"+B_ATK+"的生命\n");
            return BossHP-B_ATK;
        }
    }
}

