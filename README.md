# Muxi - Baby Care App

## 🌟 Introduction
Muxi is a **bilingual baby care tracking application** designed to help parents and caregivers seamlessly log and monitor their baby's growth and daily activities. The app provides **personalized AI insights**, supports **multi-user collaboration**, and ensures that families can easily share and track baby care information across generations.

### ✨ Features
- **Bilingual Support (English & Chinese)** 🌏
  - 实时语音输入支持
  - 双语界面切换
- **Growth & Activity Tracking** 📊
  - 喂奶记录（母乳/奶瓶）
  - 换尿布记录（类型和数量）
  - 睡眠追踪
  - 洗澡记录
  - 活动记录
- **Family Collaboration** 👨‍👩‍👧‍👦
  - 多用户数据同步
  - 权限管理
- **Daily Reminders & Notifications** 🔔 (never miss important moments!)

---

## 🛠 Tech Stack
- **Frontend**: Flutter + Riverpod
- **Database**: SQLite (Drift)
- **Key Dependencies**:
  - GoRouter: 路由管理
  - Speech-to-Text: 语音识别
  - Permission Handler: 权限管理

## 🚀 Development Setup

### **Prerequisites**
- **Flutter SDK**
- **Dart SDK**
- **Android Studio / Xcode**

### **Getting Started**

1. Clean and setup the project:
```bash
flutter clean
flutter pub get
rm -rf lib/**/*.g.dart
dart run build_runner build --delete-conflicting-outputs
flutter pub run build_runner build
flutter pub run build_runner watch
flutter gen-l10n
flutter run
```

2. Run the app on a simulator or connected device:
```bash
flutter run
```

3. Build for release:

### Android Release
```bash
# 更新版本号（在 pubspec.yaml 中修改）
# version: 1.0.0+2  # 格式：x.y.z+build_number

# 生成 release 包
flutter clean
flutter gen-l10n
flutter build appbundle  # 生成 Google Play 上传包
flutter build apk       # 生成本地安装包

# 生成的文件位置：
# AAB: build/app/outputs/bundle/release/app-release.aab
# APK: build/app/outputs/flutter-apk/app-release.apk
```
```bash
# 更新版本号（在 pubspec.yaml 中修改）
# version: 1.0.0+2  # 格式：x.y.z+build_number

# 生成 release 包
flutter clean
flutter gen-l10n
flutter build ios

# 使用 Xcode 打开项目
cd ios
open Runner.xcworkspace

# 在 Xcode 中：
# 1. 选择 Product > Archive
# 2. 在 Archives 窗口中选择最新的归档
# 3. 点击 "Distribute App"
# 4. 选择发布方式：
#    - App Store Connect（上传到 App Store）
#    - Ad Hoc（用于测试设备）
#    - Enterprise（企业内部分发）
```
---

## 📚 Documentation & Resources
- [Flutter Official Documentation](https://flutter.dev/docs)
- [Dart Language Guide](https://dart.dev/guides)
- [Muxi Project Wiki](#) *(Coming Soon!)*
- [Free icons] https://www.flaticon.com/search?word=summary

---

## 💡 Contributing
Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

---

## 📜 License
Muxi - Baby Care App is licensed under the [MIT License](LICENSE).


### **Getting Started**

1. Clean the project:
```bash
flutter clean
flutter pub get
rm -rf lib/**/*.g.dart
dart run build_runner build --delete-conflicting-outputs
flutter pub run build_runner build
flutter pub run build_runner watch
flutter gen-l10n
flutter run

# Stop the app first if it's running
rm -f ~/Library/Containers/com.example.muxi/Data/Documents/muxi.sqlite
flutter pub run build_runner build