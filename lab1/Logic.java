package lab1;



import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;



public class Logic {
    private Scanner sc = new Scanner(System.in);
    public void linear() throws IOException {
        double s = 0;
        double u = 0;
        System.out.println ("-----------Лінійний алгоритм-----------");
// Визначення джерела вводу даних
        System.out.println ("Якщо бажаєте здійснити ввід даних з клавіатури натисніть 1, якщо з файлу - 0 ");
        int t = sc.nextInt();
// Введення даних
        if (t == 1) {
            System.out.println ("Введіть значення змінної s: ");
            s = sc.nextDouble();
            System.out.println ("Введіть значення змінної u: ");

            u = sc.nextDouble();
        } if (t == 0) {
            FileInputStream fis = new FileInputStream("C:/--==-- " +
                    "/Users/Vova/IdeaProjects/mope/src/lab1/linear.txt");
            Scanner scf = new Scanner(fis);
            s = scf.nextDouble();
            u = scf.nextDouble();
        }
// Обчислення виразу
        if (s * u < 0) {
            System.out.println ("Помилка! Підлогарифмічний вираз менший за нуль");

        } if (u * s == 0) {
            System.out.println ("Помилка! &#39;u&#39; та &#39;s&#39; не можуть дорівнювати нулю");

        } else {
            double M6 = Math.log10(s / u) + Math.pow((s / u), 7);
            System.out.println ("М6 = " + M6);
        }
        System.out.println ();
    }
    public void divide() throws FileNotFoundException {
        double g = 0;
        double h = 0;
        double n = 0;
        double b = 0;
        double k = 0;
        double j = 0;
        System.out.println ("-----------Алгоритм, що розгалужується-----------");
// Визначення джерела вводу даних
        System.out.println ("Якщо бажаєте здійснити ввід даних з клавіатури, нажміть 1, якщо з файлу - 0");
        int t = sc.nextInt();
// Введення даних
        if (t == 1) {
            System.out.println ("Введіть значення змінної g:");
            g = sc.nextDouble();
            System.out.println ("Введіть значення змінної h:");
            h = sc.nextDouble();
            System.out.println ("Введіть значення змінної n:");
            n = sc.nextDouble();
            System.out.println ("Введіть значення змінної b:");
            b = sc.nextDouble();
            System.out.println ("Введіть значення змінної k:");
            k = sc.nextDouble();
            System.out.println ("Введіть значення змінної j:");
            j = sc.nextDouble();
        } if (t == 0) {
            FileInputStream fis = new FileInputStream("C:/--==-- /Users/Vova/IdeaProjects/mope/src/lab1/divide.txt");
            Scanner scf = new Scanner(fis);
            g = scf.nextDouble();
            h = scf.nextDouble();
            n = scf.nextDouble();
            b = scf.nextDouble();
            k = scf.nextDouble();
            j = scf.nextDouble();
        }
// Обчислення виразу
        if (n * b == 0) {
            System.out.println ("Помилка! Значення n та b повинні відрізнятись від нуля");

        } if ( (g > 0) || ( (g * h) % 2 == 0) ) {
            double a = Math.pow(g, g * h) / n / Math.pow(b, k * j);
            System.out.println ("a = " + a);
        } else {
            System.out.println ("Помилка! Чисельник повинен бути додатним");
        }
        System.out.println ();
    }
    public void cyclical() throws FileNotFoundException {
        double a = 0;
        double p;
        System.out.println ("-----------Циклічний алгоритм-----------");
// Визначення джерела вводу даних
        System.out.println ("Якщо бажаєте здійснити ввід даних з клавіатури, нажміть 1, якщо з файлу - 0");
        int t = sc.nextInt();
// Введення даних
        if (t == 1) {
// Обчислення виразу
            for (int i = 0; i < 25; i++) {
                System.out.println ("Введіть значення змінної a: ");
                p = sc.nextDouble();
                a += p;
            }
        } if (t == 0) {
            FileInputStream fis = new FileInputStream(":/--==-- /Users/Vova/IdeaProjects/mope/src/lab1/cyclical.txt");
            Scanner scf = new Scanner(fis);
// Обчислення виразу
            for (int i = 0; i < 25; i++) {
                p = scf.nextDouble();
                a += p;
            }
        }
        System.out.println ("Середнє арифметичне 25 введених чисел: a = " + (a / 25));

        System.out.println ();
    }
        }