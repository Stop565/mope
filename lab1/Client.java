package lab1;
import java.io.IOException;
import java.util.Scanner;
public class Client {
    public static void main(String[] args) throws IOException {
        Logic l = new Logic();
        Scanner sc = new Scanner(System.in);
        int t = 4;
        do {
            switch (t) {
                case 1:
                    l.linear();
                    t = sc.nextInt();
                    break;
                case 2:
                    l.divide();
                    t = sc.nextInt();
                    break;

                case 3:
                    l.cyclical();
                    t = sc.nextInt();
                    break;
                case 4:
                    System.out.println("******************** Меню ********************");

                    System.out.println("Введіть номер операції, яку бажаєте виконати:");

                    System.out.println("1. Виконання лінійного алгоритму;");
                    System.out.println("2. Виконання алгоритму, що розгалужується;");

                    System.out.println("3. Виконання циклічного алгоритму;");
                    System.out.println("4. Повернутись в меню;");
                    System.out.println("0. Вихід;");
                    t = sc.nextInt();
                    break;
            }
        } while (t != 0);
    }
}