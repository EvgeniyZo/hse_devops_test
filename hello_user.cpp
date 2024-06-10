#include <iostream>
#include <string>
using namespace std;

void greet_user(string user_name)
{
    cout << " Hello, " << user_name << "!";
}

int main() 
{
    string user_name;
    cin >> user_name;
    greet_user(user_name);
    return 0;
}
